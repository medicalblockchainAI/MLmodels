extern crate calamine;
extern crate rusty_machine;

use train::calamine::{open_workbook, Xlsx, Reader,DataType};
use train::rusty_machine::learning::nnet::{NeuralNet, BCECriterion};
use train::rusty_machine::learning::toolkit::regularization::Regularization;
use train::rusty_machine::learning::optim::grad_desc::StochasticGD;
use train::rusty_machine::linalg::Matrix;
use train::rusty_machine::learning::SupModel;




pub fn train() {
    let mut excel: Xlsx<_> = open_workbook("./../diabetes.xlsx").expect("Cannot open file");

    if let Some(Ok(r)) = excel.worksheet_range("diabetes") {

//        let mut data_array :Vec<f64> = Vec::new();
        let mut inputs_array :Vec<f64> = Vec::new();
        let mut targets_array :Vec<f64> = Vec::new();


        for row in r.rows().skip(1) {
            let mut counter:i32 = 0;
            for (_i, c) in row.iter().enumerate() {
                match *c {
                    DataType::Float(ref f) => {
//                        data_array.push(*f)
                        if counter == 8{
                            targets_array.push(*f)
                        }else{
                            inputs_array.push(*f)
                        }
//                        println!("conter:{:?}, value:{:?}",counter,*f);
                    },
                    _ => (),
                };
                counter = counter +1;
            };


        }
        //start NN
//        println!("inputs_array:{:?}",inputs_array);
//        println!("targets_array:{:?}",targets_array);


        let row_no = targets_array.len() as usize;
//        println!("row_no:{:?}",row_no);
        let inputs = Matrix::new(row_no,8, inputs_array);
        let targets = Matrix::new(row_no,1,  targets_array);

//        println!("inputs:{:?}",inputs);


        // Set the layer sizes - from input to output
        let layers = &[3,5,11,7,3];

        // Choose the BCE criterion with L2 regularization (`lambda=0.1`).
        let criterion = BCECriterion::new(Regularization::L2(0.1));

        // We will just use the default stochastic gradient descent.
        let mut model = NeuralNet::new(layers, criterion, StochasticGD::default());

        // rain the model!
        model.train(&inputs, &targets).unwrap();

        let test_inputs = Matrix::new(1,8, vec![1.0,93.0,70.0,31.0,0.0,30.4,0.315,23.0]);

        // And predict new output from the test inputs
        let outputs = model.predict(&test_inputs).unwrap();

        println!("output:{:?}",outputs)
    }
}