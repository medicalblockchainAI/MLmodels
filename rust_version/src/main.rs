mod train;

<<<<<<< HEAD
use train::train;


fn main() {
=======
use calamine::{open_workbook, Xlsx, Reader, DataType};
use rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use rusty_machine::learning::toolkit::regularization::Regularization;
use rusty_machine::learning::optim::grad_desc::StochasticGD;
use rusty_machine::linalg::Matrix;
use rusty_machine::learning::SupModel;

fn main() {
    let mut excel: Xlsx<_> = open_workbook("./../diabetes.xlsx").expect("Cannot open file");

    if let Some(Ok(r)) = excel.worksheet_range("diabetes") {
        for row in r.rows().skip(1) {
            let mut data_array: Vec<f64> = Vec::new();

            for (_i, c) in row.iter().enumerate() {
                match *c {
                    DataType::Float(ref f) => data_array.push(*f),
                    _ => (),
                };

                let inputs = Matrix::new(1, 8, &data_array[0..7]);
                let targets = Matrix::new(1, 1, &data_array[8..]);
            };

//                let data:&[f64] = &data_array[0..7];
//                let target = data_array[8] as i32;
//                let target:f64 = data_array[8];

//                println!("data:{:?}, target:{:?}", data, target);

            //start NN


            // Set the layer sizes - from input to output
            let layers = &[3, 5, 11, 7, 3];

            // Choose the BCE criterion with L2 regularization (`lambda=0.1`).
            let criterion = BCECriterion::new(Regularization::L2(0.1));

            // We will just use the default stochastic gradient descent.
            let mut model = NeuralNet::new(layers, criterion, StochasticGD::default());

            // rain the model!
            model.train(&inputs, &targets).unwrap();

            let test_inputs = Matrix::new(1, 8, vec![1.0, 93.0, 70.0, 31.0, 0.0, 30.4, 0.315, 23.0]);
>>>>>>> 868b3cabefff2cf00f5ce45b2c801240fab9263b

    train();

<<<<<<< HEAD

=======
            println!("output:{:?}", outputs)
        }
    }
>>>>>>> 868b3cabefff2cf00f5ce45b2c801240fab9263b
}
