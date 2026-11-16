// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// An early return runs the whole reverse sweep, which then also passes over
// the code after the return. The forward sweep did not run that code: what it
// declares holds placeholders, and dividing a zero adjoint by one gives a NaN.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

// What the code after the return declares depends on what that code writes
// first: `r` cannot be computed where the reverse sweep is defined, ahead of
// the writes to `c`.
double skipped_division(double x, double y) {
  if (y < 0)
    return x * x;
  double c[2];
  c[0] = x * y;
  c[1] = x + y;
  const double r = 1. + c[0] * c[1];
  const double q = x / r;
  return q * q;
}

// CHECK: void skipped_division_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK: int _returned0 = 0;
// CHECK: double r = 0.;
// CHECK: auto _rev0 = [&] {
// CHECK-NEXT: if (!_returned0) {
// CHECK: _returned0 = 1;
// CHECK-NEXT: _rev0();
// CHECK-NEXT: return;
// CHECK: r = 1. + c[0] * c[1];

// Each return skips a different stretch of what follows it.
double cascade(double v, double r, int order) {
  if (order == 0)
    return v * r;
  const double l = r * r - 1.;
  const double a = v / l;
  if (order == 1)
    return a;
  const double m = l * l - 9.;
  const double b = a / m;
  return b * r;
}

// CHECK: void cascade_grad_0_1(double v, double r, int order, double *_d_v, double *_d_r) {
// CHECK: if (!_returned0) {
// CHECK: if (!_returned0 || _returned0 > 1)

// Nothing follows these returns, and nothing is added for them.
double nothing_skipped(double x, double y) {
  const double z = x * y;
  if (y < 0)
    return z * x;
  else
    return z / x;
}

// CHECK: void nothing_skipped_grad(double x, double y, double *_d_x, double *_d_y) {
// CHECK-NEXT:     bool _cond0 = false;
// CHECK-NEXT:     double _d_z = 0.;
// CHECK-NEXT:     const double z = x * y;
// CHECK-NEXT:     auto _rev0 = [&] {
// CHECK-NEXT:         if (_cond0) {
// CHECK-NEXT:             _d_z += 1 * x;
// CHECK-NEXT:             *_d_x += z * 1;
// CHECK-NEXT:         } else {
// CHECK-NEXT:             _d_z += 1 / x;
// CHECK-NEXT:             double _r0 = 1 * -(z / (x * x));
// CHECK-NEXT:             *_d_x += _r0;
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             *_d_x += _d_z * y;
// CHECK-NEXT:             *_d_y += x * _d_z;
// CHECK-NEXT:         }
// CHECK-NEXT:     };
// CHECK-NEXT:     {
// CHECK-NEXT:         _cond0 = y < 0;
// CHECK-NEXT:         if (_cond0) {
// CHECK-NEXT:             _rev0();
// CHECK-NEXT:             return;
// CHECK-NEXT:         } else {
// CHECK-NEXT:             _rev0();
// CHECK-NEXT:             return;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     _rev0();
// CHECK-NEXT: }

int main() {
  auto d_skipped = clad::gradient(skipped_division);
  double dx = 0, dy = 0;
  d_skipped.execute(2., 3., &dx, &dy);
  printf("%.6f %.6f\n", dx, dy); // CHECK-EXEC: -0.001477 -0.004297
  dx = dy = 0;
  d_skipped.execute(2., -1., &dx, &dy);
  printf("%.6f %.6f\n", dx, dy); // CHECK-EXEC-NEXT: 4.000000 0.000000

  // `l` and `m` are zero where the return before them is taken.
  auto d_cascade = clad::gradient(cascade, "v, r");
  double dv = 0, dr = 0;
  d_cascade.execute(1.5, 1., 0, &dv, &dr);
  printf("%.6f %.6f\n", dv, dr); // CHECK-EXEC-NEXT: 1.000000 1.500000
  dv = dr = 0;
  d_cascade.execute(1.5, 2., 1, &dv, &dr);
  printf("%.6f %.6f\n", dv, dr); // CHECK-EXEC-NEXT: 0.333333 -0.666667
  dv = dr = 0;
  d_cascade.execute(1.5, 3., 2, &dv, &dr);
  printf("%.6f %.6f\n", dv, dr); // CHECK-EXEC-NEXT: 0.006818 -0.022113

  auto d_nothing = clad::gradient(nothing_skipped);
  dx = dy = 0;
  d_nothing.execute(2., 3., &dx, &dy);
  printf("%.6f %.6f\n", dx, dy); // CHECK-EXEC-NEXT: 0.000000 1.000000
  dx = dy = 0;
  d_nothing.execute(2., -3., &dx, &dy);
  printf("%.6f %.6f\n", dx, dy); // CHECK-EXEC-NEXT: -12.000000 4.000000
}
