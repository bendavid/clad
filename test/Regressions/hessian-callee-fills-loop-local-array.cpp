// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// The reverse pass of a hessian differentiates pushforwards that did not
// exist when the derivatives were planned, so nothing was derived ahead for
// the calls in them. A callee that overwrites an array of the loop body still
// needs its reverse_forw there, for the iterations before it to see what they
// wrote -- also when it takes integers and literals among its arguments.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double coeff(const double* p, int order, int which) {
  return p[1] * (order + 1.) + 0.1 * which * p[2];
}

void fill(const double* p, int c, int side, double L, double* I) {
  I[0] = 1. + 0.1 * c + 0.05 * side;
  const double g0 = coeff(p, 0, c);
  I[1] = g0 * L + 0.3 * c;
  if (side > 1)
    return;
  const double g1 = coeff(p, 1, c);
  I[2] = g1 * p[2] * L * L + 0.5 * g0 * g0 * L;
}

double kernel(const double* p) {
  const double as = p[0];
  double total = 0.;
  for (int c = 0; c < 2; ++c) {
    double IA[3];
    fill(p, c, 0, p[3], IA);
    total += IA[0] + as * (IA[1] + as * IA[2]);
  }
  return total;
}

// CHECK: void kernel_hessian(const double *p, double *hessianMatrix) {
// CHECK: fill_pushforward_reverse_forw(

int main() {
  auto h = clad::hessian(kernel, "p[0:3]");
  double p[4] = {0.3, 0.7, 1.1, 1.3}, H[16] = {};
  h.execute(p, H);
  for (int i = 0; i < 4; ++i)
    printf("%.5f %.5f %.5f %.5f\n", H[4 * i], H[4 * i + 1], H[4 * i + 2],
           H[4 * i + 3]);
  // CHECK-EXEC: 12.30931 8.23940 3.25546 6.84739
  // CHECK-EXEC-NEXT: 8.23940 0.23400 0.62010 1.76550
  // CHECK-EXEC-NEXT: 3.25546 0.62010 0.03159 0.74397
  // CHECK-EXEC-NEXT: 6.84739 1.76550 0.74397 0.57618
}
