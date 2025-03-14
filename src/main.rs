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

struct Hamiltonian {
    mu: f32,
    k: f32,
}

impl Hamiltonian {
    fn new(mu: f32, k: f32) -> Self {
        Self { mu, k }
    }
}

struct HamiltonianTensor<B: Backend> {
    mu: Tensor<Autodiff<B>, 1>,
    k: Tensor<Autodiff<B>, 1>,
}

impl<B: Backend> HamiltonianTensor<B> {
    fn new(mu: f32, k: f32, device: &B::Device) -> Self {
        let mu = TensorData::new(vec![mu], [1]);
        let k = TensorData::new(vec![k], [1]);
        Self {
            mu: Tensor::from_data(mu, device).require_grad(),
            k: Tensor::from_data(k, device).require_grad(),
        }
    }
}

fn compute_energy_tensor<B: Backend, const D: usize>(
    position: Tensor<Autodiff<B>, D>,
    HamiltonianTensor { mu, k }: &HamiltonianTensor<B>,
) -> Tensor<Autodiff<B>, D> {
    let position = position.detach();
    let shifted = position.clone().sub(mu.clone().expand(position.shape()));

    k.clone().expand(position.shape()).mul_scalar(0.5) * shifted.clone() * shifted
}

fn compute_energy(system: &System, hamiltonian: &Hamiltonian) -> f32 {
    let shifted = hamiltonian.mu - system.position;
    0.5 * hamiltonian.k * shifted * shifted
}

fn step(system: &mut System, hamiltonian: &Hamiltonian, rng: &mut ThreadRng) {
    let energy = compute_energy(system, hamiltonian);

    let proposed_position = system.position + rng.random_range(-1.0..1.0);

    let proposed_energy = compute_energy(
        &System {
            position: proposed_position,
        },
        hamiltonian,
    );

    let delta_energy = proposed_energy - energy;

    if delta_energy < 0.0 || rng.random_range(0.0..1.0) < (-delta_energy).exp() {
        system.position = proposed_position;
    }
}

fn data_gen() -> Result<(), Error> {
    let mut rng = rand::rng();
    let hamiltonian = Hamiltonian::new(1.0, 2.0);
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

fn train() -> Result<(), Error> {
    type B = burn::backend::NdArray;
    // type B = burn_cuda::Cuda;
    let n_epochs = 10_000;
    let batch_size = 10_000;
    let lr = 0.005;
    let device = Default::default();

    let mut mu = 1.4;
    let mut k = 1.2;

    let mut hamiltonian_tensor = HamiltonianTensor::<B>::new(mu, k, &device);
    let mut rng = rand::rng();

    // Load data
    let mut npz = NpzReader::new(std::fs::File::open("data.npz")?)?;
    let positions_data: Array1<f32> = npz.by_name("positions")?;
    let data_shape = positions_data.shape();
    let n_batches = data_shape[0] / batch_size;
    let mut positions_data: Vec<f32> = positions_data.to_vec();
    positions_data = positions_data[0..n_batches * batch_size].to_vec();
    positions_data.shuffle(&mut rng);
    let data_shape = [positions_data.len()];
    let positions_data = TensorData::new(positions_data, data_shape);
    let positions_data: Tensor<Autodiff<B>, 1> = Tensor::from_data(positions_data, &device);
    let batches = positions_data.chunk(n_batches, 0);

    let mut ks: Array1<f32> = Array1::zeros([n_epochs]);
    let mut mus: Array1<f32> = Array1::zeros([n_epochs]);

    for epoch in 0..n_epochs {
        let positions: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                let hamiltonian = Hamiltonian::new(mu, k);
                let mut system = System::new(hamiltonian.mu);

                for _ in 0..1000 {
                    step(&mut system, &hamiltonian, &mut rng);
                }
                system.position
            })
            .collect();
        let positions = TensorData::new(positions, [batch_size]);
        let positions = Tensor::<Autodiff<B>, 1>::from_data(positions, &device);

        let sim_energies = compute_energy_tensor(positions, &hamiltonian_tensor);
        let sim_energy = sim_energies.mean();

        // select a random batch
        let idx = rng.random_range(0..batches.len());
        let batch = batches[idx].clone();

        let data_energies = compute_energy_tensor(batch, &hamiltonian_tensor);
        let data_energy = data_energies.mean();

        let grads_sim = sim_energy.backward();
        let grads_data = data_energy.backward();

        let grad_mu = hamiltonian_tensor.mu.clone().grad(&grads_data).unwrap()
            - hamiltonian_tensor.mu.grad(&grads_sim).unwrap();

        let grad_k = hamiltonian_tensor.k.clone().grad(&grads_data).unwrap()
            - hamiltonian_tensor.k.grad(&grads_sim).unwrap();

        mu = (hamiltonian_tensor.mu.clone().inner() - grad_mu.mul_scalar(lr))
            .to_data()
            .to_vec()
            .unwrap()[0];
        k = (hamiltonian_tensor.k.clone().inner() - grad_k.mul_scalar(lr))
            .to_data()
            .to_vec()
            .unwrap()[0];
        hamiltonian_tensor = HamiltonianTensor::new(mu, k, &device);

        if (epoch + 1) % 100 == 0 {
            println!("epoch: {}, mu: {}, k:{}", epoch + 1, mu, k);
        }

        ks[epoch] = k;
        mus[epoch] = mu;
    }

    let mut npz = NpzWriter::new(std::fs::File::create("training.npz")?);
    npz.add_array("ks", &ks)?;
    npz.add_array("mus", &mus)?;

    Ok(())
}

fn main() -> Result<(), Error> {
    // data_gen()?;
    train()?;

    Ok(())
}
