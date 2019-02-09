import net as nn
import argparse

def get_model_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', '--learning_rate', default=0.01, help='Iniitial learning rate', type=float)
    parser.add_argument('-ep', '--epochs', default=10, help='Number of training epochs', type=int)
    parser.add_argument('-bs', '--batch_size', default=1, help='Mini-batch size', type=int)
    parser.add_argument('-dd', '--data_dir', default='/data/hdf5', help='Directory storing the training data.', type=str)
    parser.add_argument('-cd', '--chkpt_dir', default='/chkpt', help='Directory for model checkpoints', type=str)
    #parser.add_argument('-dr', '--decay_rate', default=0.8, help='Learning Rate Decay Rate')
    #parser.add_argument('-ds', '--decay_steps', default=10, help='Steps before updating learning rate')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_model_params()

    model = nn.net()
    model.create_network()
    #model.train(batch_size=args.batch_size,
    #            lr=args.learning_rate,
    #            epochs=args.epochs)
                
    
    