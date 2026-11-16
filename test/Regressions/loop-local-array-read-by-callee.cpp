// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// An array declared in a loop body without an initializer is one array in the
// reverse sweep, written again by every iteration. A call that reads it makes
// what it held to be recorded, and the declaration, which stores nothing, must
// not drop that when the analysis comes around the loop again.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double sum_squares(const double* arr, int n) {
  double r = 0;
  for (int i = 0; i < n; i++)
    r += arr[i] * arr[i];
  return r;
}

double refilled(double seed) {
  double sum = 0;
  for (int i = 0; i < 3; i++) {
    double arr[3];
    arr[0] = seed;
    arr[1] = seed * i;
    arr[2] = seed + i;
    sum += sum_squares(arr, 3);
  }
  return sum;
}

// The first iteration records what the array held before anything was
// written to it, so its clone is not left uninitialized.
// CHECK: void refilled_grad(double seed, double *_d_seed) {
// CHECK: double arr[3] = {0};
// CHECK: clad::push(_t1, arr[0]);
// CHECK-NEXT: arr[0] = seed;
// CHECK: sum_squares_pullback(arr, 3,
// CHECK: arr[0] = clad::pop(_t1);

int main() {
  auto d_refilled = clad::gradient(refilled);
  double d = 0;
  // 22 * seed + 6
  d_refilled.execute(1.5, &d);
  printf("%.2f\n", d); // CHECK-EXEC: 39.00
}
