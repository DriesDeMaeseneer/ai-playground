mod activation_functions;
mod perceptron;
use perceptron::BinaryPerceptron;

fn main() {
    let class_and_samples: Vec<(i8, [f32; 2])> = vec![
        (1, [1.0, 2.0]),
        (1, [0.0, 5.0]),
        (-1, [-1.0, -6.0]),
        (-1, [-3.0, -1.0]),
        (-1, [-5.0, 0.0]),
        (-1, [-7.0, 4.0]),
    ];
    let mut perc = BinaryPerceptron::<2>::new();
    perc.train(class_and_samples);
    println!(
        "Classified {:?} as {}, {:?}",
        [10.0, 10.0],
        perc.classify([-10.0, -10.0]),
        perc.get_weights()
    );

    println!("Hello, world!");
}
