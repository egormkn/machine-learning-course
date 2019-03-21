#include <utility>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

using namespace std;

auto random_generator = mt19937(random_device()()); // NOLINT

typedef double feature_t;

typedef int class_t;

class object {
private:
    vector<feature_t> features;
    class_t class_id;
public:
    object(vector<feature_t> features, class_t class_id) :
            features(move(features)), class_id(class_id) {}

    size_t size() const { return features.size(); }

    feature_t &operator[](size_t index) { return features[index]; }

    feature_t operator[](size_t index) const { return features[index]; }

    class_t get_class() const { return class_id; }

    vector<feature_t>::const_iterator begin() const { return features.begin(); }

    vector<feature_t>::const_iterator end() const { return features.end(); }
};

namespace normalization {

    pair<vector<double>, vector<double>> z_mean(vector<object> &train_set, size_t features_size) {
        const auto train_size = train_set.size();

        vector<double> feature_means(features_size, 0);
        for (const object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                feature_means[feature_id] += object[feature_id] / train_size;
            }
        }

        vector<double> feature_deviations(features_size, 0);
        for (const object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                feature_deviations[feature_id] +=
                        pow(object[feature_id] - feature_means[feature_id], 2) / (train_size - 1);
            }
        }
        for (double &deviation : feature_deviations) {
            deviation = sqrt(deviation);
        }

        for (object &object : train_set) {
            for (int feature_id = 0; feature_id < features_size; ++feature_id) {
                object[feature_id] -= feature_means[feature_id];
                object[feature_id] /= feature_deviations[feature_id];
            }
        }
        return make_pair(feature_means, feature_deviations);
    }
}

class neuron {
private:
    vector<double> weights;
    double bias = 0.0;
    double delta = 0.0;
    double output = 0.0;

public:
    explicit neuron(size_t size) {
        uniform_real_distribution<double> distribution(-1.0 / (2 * size), 1.0 / (2 * size));
        weights.resize(size);
        for (double &weight : weights) {
            weight = distribution(random_generator);
        }
        bias = distribution(random_generator);
    }

    double activate(const vector<double> &input) {
        output = tanh(inner_product(input.begin(), input.end(), weights.begin(), bias));
        return output;
    }

    size_t size() const { return weights.size(); }

    double &operator[](size_t index) { return weights[index]; }

    double operator[](size_t index) const { return weights[index]; }

    vector<double>::const_iterator begin() const { return weights.begin(); }

    vector<double>::const_iterator end() const { return weights.end(); }

    double get_bias() const {
        return bias;
    }

    void set_bias(double bias) {
        neuron::bias = bias;
    }

    double get_delta() const {
        return delta;
    }

    void set_delta(double delta) {
        neuron::delta = delta;
    }

    double get_output() const {
        return output;
    }
};

class deep_network {
public:
    static unique_ptr<deep_network>
    make_network(const vector<object> &train_set, const vector<unsigned> &layers_size, size_t features_size) {
        const auto &objects_size = train_set.size();
        const auto &network_size = layers_size.size();

        // Make a copy of training set
        vector<object> objects = train_set;

        // Create network layers
        vector<vector<neuron>> layers;
        for (size_t i = 0, input_size = features_size; i < network_size; ++i) {
            vector<neuron> layer;
            layer.reserve(layers_size[i]);
            for (int j = 0; j < layers_size[i]; j++) {
                layer.emplace_back(input_size);
            }
            input_size = layer.size();
            layers.push_back(layer);
        }

        // Build a network
        auto network = make_unique<deep_network>(layers);

        // Train network
        uniform_int_distribution<size_t> distribution(0, objects.size() - 1);
        clock_t start_time = clock();
        for (size_t object_id = distribution(random_generator);
             clock() - start_time < 9500; object_id = distribution(random_generator)) {
            const object &object = objects[object_id];
            double output = network->train_forward(object);
            network->train_backward(output, object.get_class());
            network->correct_weights(object);
        }

        return network;
    }

    const vector<vector<neuron>> &get_layers() const {
        return layers;
    }

    explicit deep_network(vector<vector<neuron>> layers) : layers(move(layers)) {}

private:
    static constexpr double learning_rate = 0.01;

    vector<vector<neuron>> layers;

    double train_forward(const object &object) {
        vector<double> input(object.begin(), object.end()), output;

        for (vector<neuron> &layer : layers) {
            output.clear();
            for (neuron &neuron : layer) {
                output.push_back(neuron.activate(input));
            }
            input = output;
        }

        return output.front(); // Return output of last neuron
    }

    void train_backward(double output, int correct) {
        neuron &last = layers.back().front();
        last.set_delta((correct - output) * d_tanh(output));

        for (size_t i = layers.size() - 1; i > 0; --i) {
            vector<neuron> &prev_layer = layers[i - 1];
            vector<neuron> &next_layer = layers[i];
            for (int j = 0; j < prev_layer.size(); j++) {
                neuron &prev_neuron = prev_layer[j];
                double delta = 0.0;
                for (neuron &next_neuron : next_layer) {
                    delta += next_neuron.get_delta() * next_neuron[j];
                }
                prev_neuron.set_delta(delta * d_tanh(prev_neuron.get_output()));
            }
        }
    }

    inline double d_tanh(double x) {
        return 1 - pow(x, 2);
    }

    void correct_weights(const object &object) {
        for (int i = 0; i < layers.size(); i++) {
            vector<double> input;
            if (i == 0) {
                for (double feature : object) {
                    input.push_back(feature);
                }
            } else {
                for (neuron &n : layers[i - 1]) {
                    input.push_back(n.get_output());
                }
            }

            for (neuron &neuron : layers[i]) {
                for (size_t j = 0; j < input.size(); j++) {
                    neuron[j] += learning_rate * neuron.get_delta() * input[j];
                }
                neuron.set_bias(neuron.get_bias() + learning_rate * neuron.get_delta());
            }
        }
    }
};

void solve() {
    /**
     * d - number of layers including input layer
     * m - number of features
     */
    unsigned network_size, features_size;
    cin >> network_size >> features_size;

    --network_size; // Remove input layer from total number of layers

    vector<unsigned> layers_size(network_size);
    for (unsigned &size : layers_size) {
        cin >> size;
    }

    size_t train_size;
    cin >> train_size;

    vector<object> train_set;

    // Read training set
    for (int object_id = 0; object_id < train_size; ++object_id) {
        vector<feature_t> features(features_size);
        for (feature_t &feature : features) {
            cin >> feature;
        }
        class_t class_id;
        cin >> class_id;
        train_set.emplace_back(features, class_id);
    }

    // Normalize features
    auto normalization_data = normalization::z_mean(train_set, features_size);
    vector<double> feature_means = normalization_data.first;
    vector<double> feature_deviations = normalization_data.second;

    auto network = deep_network::make_network(train_set, layers_size, features_size);
    const vector<vector<neuron>> &layers = network->get_layers();

    vector<neuron> first = layers.front();
    for (neuron &n : first) {
        double bias = 0.0;
        for (size_t i = 0; i < n.size(); i++) {
            bias += feature_means[i] * n[i] / feature_deviations[i];
            cout << n[i] / feature_deviations[i] << ' ';
        }

        cout << n.get_bias() - bias << endl;
    }

    for (int i = 1; i < layers.size(); ++i) {
        for (const neuron &neuron : layers[i]) {
            for (double w : neuron) {
                cout << w << ' ';
            }

            cout << neuron.get_bias() << endl;
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

#ifdef DEBUG
    ifstream input("input.txt");
    ofstream output("output.txt");
    streambuf *cin_buffer(cin.rdbuf());
    streambuf *cout_buffer(cout.rdbuf());
    cin.rdbuf(input.rdbuf());
    cout.rdbuf(output.rdbuf());
#else
    streambuf *cerr_buffer(cerr.rdbuf());
    cerr.rdbuf(nullptr);
#endif

    cout << fixed;
    solve();
    cout.flush();

#ifdef DEBUG
    cin.rdbuf(cin_buffer);
    cout.rdbuf(cout_buffer);
    input.close();
    output.close();
#else
    cerr.rdbuf(cerr_buffer);
#endif
    return 0;
}
