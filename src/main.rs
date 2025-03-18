use anyhow::Error;
use burn::prelude::Backend;
use burn::prelude::Tensor;
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use ndarray::Array1;
use ndarray_npy::NpzReader;
use ndarray_npy::NpzWriter;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

struct System {
    position: f32,
}

impl System {
    fn new(position: f32) -> Self {
        Self { position }
    }
}

/// Bare struct with parameters for efficient MC sampling
struct Hamiltonian {
    mu: f32,
    k: f32,
    alpha: f32,
}

impl Hamiltonian {
    fn new(mu: f32, k: f32, alpha: f32) -> Self {
        Self { mu, k, alpha }
    }
}

/// Struct holding differentiable parameter tensor
struct HamiltonianTensor<B: Backend> {
    parameters: Tensor<Autodiff<B>, 1>,
}

impl<B: Backend> HamiltonianTensor<B> {
    fn new(mu: f32, k: f32, alpha: f32, device: &B::Device) -> Self {
        let parameters = TensorData::new(vec![mu, k, alpha], [3]);
        Self {
            parameters: Tensor::from_data(parameters, device).require_grad(),
        }
    }
}

/// Compute the energy of the system given the position and the Hamiltonian parameters
///
/// This energy computation supports taking the gradient with respect to the parameters.o
///
/// * `position` - The position of the particle
/// * `HamiltonianTensor` - The Hamiltonian parameters
fn compute_energy_tensor<B: Backend, const D: usize>(
    position: Tensor<Autodiff<B>, D>,
    HamiltonianTensor { parameters }: &HamiltonianTensor<B>,
) -> Tensor<Autodiff<B>, D> {
    let position = position.detach();
    let mu = parameters.clone().slice([0..1]);
    let k = parameters.clone().slice([1..2]);
    let alpha = parameters.clone().slice([2..3]);
    let shifted = position.clone().sub(mu.clone().expand(position.shape()));
    let shifted = shifted.abs();

    k.clone().expand(position.shape()) * shifted.powf(alpha.expand(position.shape()))
}

/// Compute the energy given the system and the Hamiltonian parameters
///
/// This function does not support taking the gradient with respect to the parameters to increase
/// efficiency during MC sampling.
///
/// * `system` - The system state
/// * `hamiltonian` - The Hamiltonian parameters
fn compute_energy(system: &System, hamiltonian: &Hamiltonian) -> f32 {
    let shifted = hamiltonian.mu - system.position;
    let shifted = shifted.abs();
    hamiltonian.k * shifted.powf(hamiltonian.alpha)
}

/// Single MCMC step
///
/// * `system` - The system state
/// * ``hamiltonian` - The Hamiltonian parameters
fn step(system: &mut System, hamiltonian: &Hamiltonian, rng: &mut ThreadRng) {
    let energy = compute_energy(system, hamiltonian);

    // Note: The step size should probably be chosen more carefully
    let proposed_position = system.position + rng.random_range(-1.0..1.0);

    let proposed_energy = compute_energy(
        &System {
            position: proposed_position,
        },
        hamiltonian,
    );

    let delta_energy = proposed_energy - energy;

    // Metropolis-Hastings step
    if delta_energy < 0.0 || rng.random_range(0.0..1.0) < (-delta_energy).exp() {
        system.position = proposed_position;
    }
}

/// Generate the training data
fn data_gen() -> Result<(), Error> {
    let mut rng = rand::rng();
    let hamiltonian = Hamiltonian::new(1.0, 2.0, 4.0);
    let mut system = System::new(hamiltonian.mu);

    const N_SAMPLES: usize = 1_000_000;

    let mut samples: Array1<f32> = Array1::zeros(N_SAMPLES);

    for i in 0..N_SAMPLES {
        for _ in 0..1000 {
            step(&mut system, &hamiltonian, &mut rng);
        }
        samples[i] = system.position;
    }

    let mut npz = NpzWriter::new(std::fs::File::create("data.npz")?);
    npz.add_array("positions", &samples)?;

    Ok(())
}

/// Train the model
fn train() -> Result<(), Error> {
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::NdArray;

    #[cfg(feature = "cuda")]
    type B = burn_cuda::Cuda;

    let n_epochs = 10_000;
    let batch_size = 10_000;
    let lr = 0.05;
    let device = Default::default();

    let mut mu = 1.4;
    let mut k = 1.2;
    let mut alpha = 3.0;

    let mut rng = rand::rng();

    // Load data
    let batches = load_batched_training_data(batch_size, &device, &mut rng)?;

    // Data logging
    let mut ks: Array1<f32> = Array1::zeros([n_epochs]);
    let mut mus: Array1<f32> = Array1::zeros([n_epochs]);
    let mut alphas: Array1<f32> = Array1::zeros([n_epochs]);

    // Training loop
    for epoch in 0..n_epochs {
        // sample energies with MCMC
        let positions: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                let hamiltonian = Hamiltonian::new(mu, k, alpha);
                let mut system = System::new(hamiltonian.mu);

                for _ in 0..1000 {
                    step(&mut system, &hamiltonian, &mut rng);
                }
                system.position
            })
            .collect();
        let positions = TensorData::new(positions, [batch_size]);
        let positions = Tensor::<Autodiff<B>, 1>::from_data(positions, &device);

        let hamiltonian_tensor = HamiltonianTensor::<B>::new(mu, k, alpha, &device);
        let sim_energies = compute_energy_tensor(positions, &hamiltonian_tensor);
        let sim_energy = sim_energies.mean();

        // sample energies from data
        // select a random batch
        let idx = rng.random_range(0..batches.len());
        let batch = batches[idx].clone();

        let data_energies = compute_energy_tensor(batch, &hamiltonian_tensor);
        let data_energy = data_energies.mean();

        // Gradient of loss (data - simulated) w.r.t. parameters
        let grads_sim = sim_energy.backward();
        let grads_data = data_energy.backward();

        let grad_parameters = hamiltonian_tensor
            .parameters
            .clone()
            .grad(&grads_data)
            .unwrap()
            - hamiltonian_tensor.parameters.grad(&grads_sim).unwrap();

        // stochastic gradient descent update step
        let parameters_new: Vec<f32> = (hamiltonian_tensor.parameters.clone().inner()
            - grad_parameters.mul_scalar(lr))
        .to_data()
        .to_vec()
        .unwrap();

        // Move parameters out of tensor to use them in MCMC sampler
        mu = parameters_new[0];
        k = parameters_new[1];

        // only update alpha if it isn't NaN
        if parameters_new[2].is_finite() {
            alpha = parameters_new[2];
        }

        // Logging
        if (epoch + 1) % 100 == 0 {
            println!(
                "epoch: {}, mu: {}, k:{}, alpha: {}",
                epoch + 1,
                mu,
                k,
                alpha
            );
        }
        ks[epoch] = k;
        mus[epoch] = mu;
        alphas[epoch] = alpha;
    }

    // Save weight data per epoch
    let mut npz = NpzWriter::new(std::fs::File::create("training.npz")?);
    npz.add_array("ks", &ks)?;
    npz.add_array("mus", &mus)?;
    npz.add_array("alphas", &alphas)?;

    Ok(())
}

/// Load training data in batches
///
/// * `batch_size` - The size of the batches
/// * `device` - The device to load the data on
/// * `rng` - The random number generator for shuffling
fn load_batched_training_data<B: Backend>(
    batch_size: usize,
    device: &B::Device,
    rng: &mut ThreadRng,
) -> Result<Vec<Tensor<Autodiff<B>, 1>>, Error> {
    let mut npz = NpzReader::new(std::fs::File::open("data.npz")?)?;
    let positions_data: Array1<f32> = npz.by_name("positions")?;
    let data_shape = positions_data.shape();
    let n_batches = data_shape[0] / batch_size;
    let mut positions_data: Vec<f32> = positions_data.to_vec();
    positions_data = positions_data[0..n_batches * batch_size].to_vec();
    positions_data.shuffle(rng);
    let data_shape = [positions_data.len()];
    let positions_data = TensorData::new(positions_data, data_shape);
    let positions_data: Tensor<Autodiff<B>, 1> = Tensor::from_data(positions_data, device);
    let batches = positions_data.chunk(n_batches, 0);
    Ok(batches)
}

fn main() -> Result<(), Error> {
    data_gen()?; // uncomment to generate new data
    train()?;

    Ok(())
}
