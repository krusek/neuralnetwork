//
//  LinearAlgebra.swift
//  NeuralNetwork
//
//  Created by Korben Rusek on 1/13/20.
//  Copyright © 2020 Korben Rusek. All rights reserved.
//

import Foundation

public protocol Field: Equatable {
    static func * (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func random() -> Self
    func exponent() -> Self
    static var zero: Self { get }
    static var one: Self { get }
}

extension Double: Field {
    static public func random() -> Double {
        return Double.random(in: 0...1)
    }

    public func exponent() -> Double {
        return exp(self)
    }

    public static let zero: Double = 0.0
    public static var one: Double = 1.0
}

//public typealias Scalar = Double

public struct Vector<Scalar: Field>: Equatable {
    let value: [Scalar]

    public init(value: [Scalar]) {
        self.value = value
    }

    func map<T>(_ transform: (Scalar) -> T) -> [T] {
        return value.map(transform)
    }

    func map(_ transform: (Scalar) -> Scalar) -> Vector<Scalar> {
        return Vector(value: value.map(transform))
    }

    public static func +(lhs: Self, rhs: Self) -> Self {
        return Vector(value: zip(lhs.value, rhs.value).map { $0 + $1 })
    }

    static func randn(_ d0: Int) -> Vector {
        return Vector(value: (0..<d0).map { _ in Scalar.random() })

    }

    public static func ==(lhs: Vector, rhs: Vector) -> Bool {
        return lhs.value == rhs.value
    }
}

public struct Matrix<Scalar: Field>: Equatable {
    let value: [[Scalar]]

    init(_ value: [Vector<Scalar>]) {
        let v = value.map { $0.value }
        self.value = v
    }

    public init(value: [[Scalar]]) {
        self.value = value
    }

    func map<T>(_ transform: ([Scalar]) -> T) -> [T] {
        return value.map(transform)
    }

    func map(_ transform: (Vector<Scalar>) -> Vector<Scalar>) -> Matrix {
        let vectors = value.map { (array) -> Vector<Scalar> in
            let vector = Vector(value: array)
            return transform(vector)
        }
        return Matrix(vectors)
    }

    static func randn(_ d0: Int, _ d1: Int) -> Matrix {
        return Matrix(value: (0..<d0).map { _ in (0..<d1).map { _ in Scalar.random() } })
    }

    public static func ==(lhs: Matrix, rhs: Matrix) -> Bool {
        return lhs.value == rhs.value
    }
}

public func /<Scalar: Field>(lhs: Scalar, rhs: Vector<Scalar>) -> Vector<Scalar> {
    return rhs.map { lhs / $0 }
}

func exp<Scalar: Field>(_ list: Vector<Scalar>) -> Vector<Scalar> {
    return list.map { (value) -> Scalar in
        return value.exponent()
    }
}

public func +<Scalar: Field>(lhs: Scalar, rhs: Vector<Scalar>) -> Vector<Scalar> {
    return rhs.map { lhs + $0 }
}

public func dot<Scalar: Field>(_ lhs: Vector<Scalar>, _ rhs: Vector<Scalar>) -> Scalar {
    return zip(lhs.value, rhs.value).map { $0 * $1 }.reduce(Scalar.zero, +)
}

public func *<Scalar: Field>(_ lhs: Matrix<Scalar>, _ rhs: Vector<Scalar>) -> Vector<Scalar> {
    return Vector(value: lhs.map { dot(Vector(value: $0), rhs) })
}
