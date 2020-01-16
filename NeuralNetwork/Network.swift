//
//  sigmoid.swift
//  neuralnetwork
//
//  Created by Korben Rusek on 1/10/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import Foundation

public func sigmoid<Scalar: Field>(_ list: Vector<Scalar>) -> Vector<Scalar> {
    return Scalar.one / (Scalar.one + exp(list))
}

public func sigmoidPrime<Scalar: Field>(_ list: Vector<Scalar>) -> Vector<Scalar> {
    let diff = Scalar.one - sigmoid(list)
    return sigmoid(list) * diff
}

public typealias PhotoData = [Int]
public typealias Output<Scalar: Field> = [Scalar]
public typealias TestData<Scalar: Field> = [(PhotoData, Output<Scalar>)]

public class Network<Scalar: Field> {
    let sizes: [Int]
    let biases: [Vector<Scalar>]
    let weights: [Matrix<Scalar>]
    init(sizes: [Int]) {
        self.sizes = sizes
        self.biases = sizes.dropFirst().map { Vector<Scalar>.randn($0) }
        self.weights = zip(sizes.dropLast(), sizes.dropFirst()).map { Matrix<Scalar>.randn($0, $1) }
    }

    init(sizes: [Int], biases: [Vector<Scalar>], weights: [Matrix<Scalar>]) {
        self.sizes = sizes
        self.biases = biases
        self.weights = weights
    }

    public func feedForward(_ a: Vector<Scalar>) -> Vector<Scalar> {
        return zip(biases, weights).reduce(a) { (a, arg1) -> Vector<Scalar> in
            let (b, w) = arg1

            let prod = w * a
            return sigmoid(prod + b)
        }
    }

    public func SGD(trainingData: [(PhotoData, Output<Scalar>)], epochs: Int, miniBatchSize: Int, learningRate: Scalar, testData: TestData<Scalar>?) {
        let n = trainingData.count
        var network = self
        for j in 0..<epochs {
            let data: [(PhotoData, Output)] = trainingData.shuffled()
            let ranges = (0..<n/miniBatchSize).map { (ix) -> Range<Int> in
                return (ix..<(min(ix + miniBatchSize, n)))
            }
            for range in ranges {
                network = network.updateMiniBatch(data[range], learningRate: learningRate)
            }
            if let testData = testData {
                print("Epoch \(j): 0 / \(testData.count)")
            } else {
                print("Epoch \(j) complete")
            }
        }
    }

    private func updateMiniBatch(_ batch: ArraySlice<(PhotoData, Output<Scalar>)>, learningRate: Scalar) -> Network<Scalar> {
        var nabla_b = self.biases.map { Vector<Scalar>.zero($0) }
        var nabla_w = self.weights.map { Matrix<Scalar>.zero($0) }
        for (x, y) in batch {
            let (delta_b, delta_w) = self.backprop(x, y)
            nabla_b = zip(nabla_b, delta_b).map { $0 + $1 }
            nabla_w = zip(nabla_w, delta_w).map { $0 + $1 }
        }
        let weights = zip(self.weights, nabla_w).map { (arg) -> Matrix<Scalar> in
            let (w, nw) = arg
            return w - (learningRate / batch.count) * nw
        }
        let biases = zip(self.biases, nabla_b).map { (arg) -> Vector<Scalar> in
            let (b, nb) = arg
            return b - (learningRate / batch.count) * nb
        }
        return Network(sizes: self.sizes, biases: biases, weights: weights)
    }

    private func backprop(_ data: PhotoData, _ expected: Output<Scalar>) -> ([Vector<Scalar>], [Matrix<Scalar>]) {
        var nabla_b = self.biases.map { Vector<Scalar>.zero($0) }
        var nabla_w = self.weights.map { Matrix<Scalar>.zero($0) }
        var activation = Vector<Scalar>(value: data.map { Scalar($0) })
        var activations: [Vector<Scalar>] = [activation]
        var zs: [Vector<Scalar>] = []
        for (b, w) in zip(biases, weights) {
            let z = w * activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        }

        var delta = self.costDerivative(activation: activations.last!, expected: expected) * sigmoidPrime(zs.last!)
        nabla_b[nabla_b.count - 1] = delta
        nabla_w[nabla_w.count - 1] = matrix(delta, activations.last!)
        for l in 2..<self.sizes.count {
            let z = zs.last!
            let sp = sigmoidPrime(z)
            let index = (self.weights.count - l + 1) % self.weights.count
            delta = (self.weights[index].transpose() * delta) * sp
            nabla_b[nabla_b.count - 1] = delta
            nabla_w[nabla_w.count - 1] = matrix(delta, activations[index + 1])
        }
        return (nabla_b, nabla_w)
    }

    private func matrix(_ lhs: Vector<Scalar>, _ rhs: Vector<Scalar>) -> Matrix<Scalar> {
        let value = lhs.value.map { $0 * rhs }
        return Matrix<Scalar>(value)
    }

    private func costDerivative(activation: Vector<Scalar>, expected: Output<Scalar>) -> Vector<Scalar> {
        return activation - Vector<Scalar>(value: expected)
    }
}
