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

public class Network<Scalar: Field> {
    let sizes: [Int]
    let biases: [Vector<Scalar>]
    let weights: [Matrix<Scalar>]
    init(sizes: [Int]) {
        self.sizes = sizes
        self.biases = sizes.dropFirst().map { Vector<Scalar>.randn($0) }
        self.weights = zip(sizes.dropLast(), sizes.dropFirst()).map { Matrix<Scalar>.randn($0, $1) }
    }

    public func feedForward(_ a: Vector<Scalar>) -> Vector<Scalar> {
        return zip(biases, weights).reduce(a) { (a, arg1) -> Vector<Scalar> in
            let (b, w) = arg1

            let prod = w * a
            return sigmoid(prod + b)
        }
    }
}
