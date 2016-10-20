import NN, data_loader, perceptron, csv
import matplotlib.pyplot as plt

# Parameter Settings
domain = 'mnist'
batch_sizes = [10, 50, 100]
activation_functions = ['tanh', 'relu']
learning_rates = [0.1, 0.01]
hidden_layer_widths = [10, 50]

# Loading Data
if domain == 'mnist':
    training_data, test_data = data_loader.load_mnist_data()
elif domain == 'circles':
    training_data, test_data = data_loader.load_circle_data()
else:
    raise Exception('Domain must be one of [mnist, circles]')
data_dim = len(training_data[0][0])

# Tuning Parameters
with open(domain+'.csv', 'w') as csvfile:
    fieldnames = ['batch_size', 'activation_function', 'learning_rate', 'hidden_layer_width', 'average_accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    print "\n>> Tuning Parameters"
    best_acc = 0
    best_b = -1
    best_a = -1
    best_l = -1
    best_h = -1
    for b in batch_sizes:
        for a in activation_functions:
            for l in learning_rates:
                for h in hidden_layer_widths:
                    aver_acc = 0
                    for f in range(0, 5):
                        length = len(training_data)
                        step = int(round(length/5))
                        test = training_data[f*step:(f+1)*step]
                        train = training_data[:f*step] + training_data[(f+1)*step:]

                        net = NN.create_NN(domain, b, l, a, h)
                        print net.train(train), "<- train"
                        eva_acc = net.evaluate(test)
                        print eva_acc, "<- evaluate "
                        aver_acc += eva_acc

                    aver_acc /= 5
                    print ("Parameters: b=%s, a=%s, l=%s, h=%s" % (b, a, l, h))
                    print "Average Accuracy: ", aver_acc
                    writer.writerow({
                        'batch_size': b,
                        'activation_function': a,
                        'learning_rate': l,
                        'hidden_layer_width': h,
                        'average_accuracy': aver_acc})

                    if aver_acc > best_acc:
                        best_acc = aver_acc
                        best_b = b
                        best_a = a
                        best_l = l
                        best_h = h
                        print "* Parameters updated. *"
    print ("\n>>> Best Parameters: b=%s, a=%s, l=%s, h=%s\n" % (best_b, best_a, best_l, best_h))
    csvfile.close()

print "\n>> Learning Curves"
net = NN.create_NN(domain, best_b, best_l, best_a, best_h)
net_curve = net.train_with_learning_curve(training_data)
print net_curve

perc = perceptron.Perceptron(data_dim)
perc_curve = perc.train_with_learning_curve(training_data)
print perc_curve

# Plot learning curves
n_net, net_acc = zip(*net_curve)
n_perc, perc_acc = zip(*perc_curve)
perc_acc = map(lambda x: x * 100, perc_acc)
print perc_acc
plt.plot(n_net, net_acc, n_perc, perc_acc)
plt.legend(('Neural Network', 'Perceptron'), loc='best')
plt.title('Learning Curves between NN and Perceptron')
plt.show()
