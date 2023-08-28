use std::{marker::PhantomData, ptr::null};

use halo2_proofs::{plonk::*, arithmetic::Field, halo2curves::{FieldExt, pasta::{pallas, Fp}}, circuit::{Layouter, AssignedCell, Value, SimpleFloorPlanner}};
use halo2_gadgets::{poseidon::Pow5Chip as PoseidonChip, poseidon::{Pow5Config as PoseidonConfig, primitives::{P128Pow5T3, Spec, ConstantLength}, Hash as PoseidonHash}, utilities::FieldValue};

pub const WIDTH: usize = 3;
pub const RATE: usize = 2;
pub const MESSAGE_SIZE: usize = 2;

#[derive(Clone, Debug)]
struct Config {
    advices: [Column<Advice>; 3],
    instance: Column<Instance>,
    poseidon: PoseidonConfig<pallas::Base, WIDTH, RATE>,
}

#[derive(Debug, Clone)]
struct FunctionChip {
    config: Config,
}

impl FunctionChip {
    pub fn construct(config: Config) -> Self {
        Self {config}
    }

    pub fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Config {
        let output = meta.instance_column();
        meta.enable_equality(output);

        // let state = (0..WIDTH).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let state = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let partial_sbox = meta.advice_column();

        let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

        meta.enable_constant(rc_b[0]);

        let poseidon = PoseidonChip::configure::<P128Pow5T3>(
            meta, state.try_into().unwrap(), partial_sbox, rc_a.try_into().unwrap(), rc_b.try_into().unwrap()
        );

        println!("configure is done");

        Config { advices: state, instance: output, poseidon }
    }

    pub fn assign(
        &self,
        elem_1: Value<Fp>,
        elem_2: Value<Fp>,
        output_in: Value<Fp>,
        mut layouter: impl Layouter<pallas::Base>
    ) -> Result<AssignedCell<pallas::Base, pallas::Base>, Error>
    {        

        let poseidon_chip = PoseidonChip::construct(self.config.poseidon.clone());

        let note = layouter.assign_region(
            || "output",
            |mut region| {

                let secret = region.assign_advice(|| "elem_1", self.config.advices[0] , 0, || elem_1)?;
                let nullifier = region.assign_advice(|| "elem_2", self.config.advices[0] , 0, || elem_2)?;
                Ok([secret, nullifier])
            }
        )?;

        let hasher = PoseidonHash::<_, _, P128Pow5T3, ConstantLength<2>, WIDTH, RATE>::init(
            poseidon_chip,
            layouter.namespace(|| "init"),
        )?;
        let output = hasher.hash(layouter.namespace(|| "hash"), note)?;

        println!("hash: {:?}", output.cell().clone());

        // let output = self.hash(self.config.clone(), layouter.namespace(|| "assign output"), note)?;
        layouter.assign_region(
            || "constrain output",
            |mut region| {
                let expected_var = region.assign_advice(
                    || "load output",
                    self.config.advices[0],
                    0,
                    || output_in,
                )?;
                region.constrain_equal(output.cell(), expected_var.cell())
            },
        )?;

        println!("assign is done");

        Ok(output)
    
    }

    pub fn expose_public<F: Field> (
        &self,
        mut layouter: impl Layouter<F>,
        cell: &AssignedCell<F, F>,
        row: usize,
     ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }
}

#[derive(Default)]
struct HashCircuit {
    secret: Value<Fp>,
    nullifier: Value<Fp>,
    output: Value<Fp>,
}

impl Circuit<pallas::Base> for HashCircuit {
    type Config = Config;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        Self::configure(meta)
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<pallas::Base>) -> Result<(), Error> {
        let chip = FunctionChip::construct(config);
        let output_cell = chip.assign(self.secret, self.nullifier, self.output, layouter.namespace(|| "next row"))?;
        chip.expose_public(layouter.namespace(|| "expose hash"), &output_cell, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use halo2_gadgets::poseidon::{primitives::P128Pow5T3 as OrchardNullifier};
    use halo2_proofs::{dev::MockProver};
    use halo2_gadgets::poseidon::primitives::Hash;
    use super::*;
    #[test]
    fn test_fib() {

        let message = [Fp::from(2), Fp::from(3)];
        // let output =
        let poseidon_hasher = Hash::<_, OrchardNullifier, ConstantLength<2>, 3, 2>::init().hash(message);
        println!("poseido_hasher: {:?}", poseidon_hasher);

        let circuit = HashCircuit {
            secret: Value::known(Fp::from(2)),
            nullifier: Value::known(Fp::from(3)),
            output:Value::known(poseidon_hasher),
        };

        
        let mut public_input = vec![poseidon_hasher];
        let prover = MockProver::run(18, &circuit, vec![public_input]).unwrap();
        prover.assert_satisfied();
    }

}