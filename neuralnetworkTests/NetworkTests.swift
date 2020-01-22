//
//  NetworkTests.swift
//  NeuralNetworkTests
//
//  Created by Korben Rusek on 1/21/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import XCTest
import NeuralNetwork

func ==(lhs: Vector<Double>, rhs: [Double]) -> Bool {
    return lhs == Vector<Double>(value: rhs)
}

class NetworkTests: XCTestCase {
    func testZeros() {
        let zeroGenerator = {
            return Double(0)
        }
        let net = Network<Double>(sizes: [784, 30, 10], generator: zeroGenerator)
        let value = net.feedForward(Vector<Double>(value: (0..<784).map{ Double($0) }))
        XCTAssertEqual(value, Vector<Double>(value: Array(repeating: 0.5, count: 10)))
    }

    func testSimple() {
        let generator = {
            Double(1)
        }
        let net = Network<Double>(sizes: [3, 2, 1], generator: generator)
        let value = net.feedForward(Vector<Double>(value: [1.0, 2.0, 1.0]))
        XCTAssertEqual(value, Vector<Double>(value: [0.2654198479885073]))
    }

    func testSDG() throws {
        let net = Network<Double>(sizes: [784, 30, 10])
        let training = try Stream.readTrainingData(.train)
        let test = try Stream.readTrainingData(.t10k)
        net.SGD(trainingData: training, epochs: 30, miniBatchSize: 10, learningRate: 3, testData: test)
    }
}
