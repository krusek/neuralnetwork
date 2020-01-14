//
//  LinearAlgebraTests.swift
//  neuralnetworkTests
//
//  Created by Korben Rusek on 1/13/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import XCTest
import NeuralNetwork

class LinearAlgebraTests: XCTestCase {
    func testExample() {
        let vector = Vector(value: [1.0, 2.0, 3.0])
        let matrix = Matrix(value: [
            [0.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [2.0, -3.0, 1.0]
        ])
        XCTAssertEqual(Vector(value: [5.0, 2.0, -1.0]), matrix * vector)
    }
}
