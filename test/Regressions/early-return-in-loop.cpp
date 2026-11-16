// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A return inside a loop leaves in the middle of an iteration: the rest of
// the body and the step to the next iteration were not run for it, the
// iterations before it ran in full. Every derivative is compared with central
// differences, for every way out of the function.

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

double in_for(double x, int n) {
  double s = 0.;
  for (int i = 0; i < 4; ++i) {
    double t = x * (i + 1.);
    if (i == n)
      return s + t * t;
    double r = t + 1.;
    s += x / r;
  }
  return s;
}

// CHECK: void in_for_grad_0(double x, int n, double *_d_x) {
// CHECK: for (; _t0; _t0--) {
// CHECK-NEXT: if (!_returned0)
// CHECK-NEXT: --i;
// CHECK: _returned0 = 0;

// Only the increment follows the return.
double last_in_body(double x, int n) {
  double s = 1.;
  for (int i = 0; i < 4; ++i) {
    s *= x + 0.5 * i;
    if (i == n)
      return s * s;
  }
  return s;
}

double nested(double x, int n) {
  double s = 0.;
  for (int i = 0; i < 3; ++i) {
    s += x;
    for (int j = 0; j < 2; ++j) {
      double t = x * (j + 1.) + s;
      if (i * 2 + j == n)
        return s * t;
      double r = t + 1.;
      s += x * j / r;
    }
    s *= 1. + 0.1 * i * x;
  }
  return s;
}

double two_returns(double x, int n) {
  double s = 0.;
  for (int i = 0; i < 4; ++i) {
    double a = x * x + i;
    if (2 * i == n)
      return s + a;
    double b = a / (x + 2.);
    s += b;
    if (2 * i + 1 == n)
      return s * b;
    double c = b * b + 1.;
    s += x / c;
  }
  double z = s * s + 1.;
  return x / z;
}

double in_while(double x, int n) {
  double s = 0.;
  int i = 0;
  while (i < 4) {
    double t = x * (i + 1.);
    if (i == n)
      return s + t * t;
    double r = t + 1.;
    s += x / r;
    ++i;
  }
  double w = s + 2.;
  return x / w;
}

double in_range_for(double x, int n) {
  double w[4] = {1., 2., 3., 4.};
  double s = 0.;
  int i = 0;
  for (double wi : w) {
    double t = x * wi;
    if (i == n)
      return s + t * t;
    double r = t + 1.;
    s += x / r;
    ++i;
  }
  return s;
}

// Returns before, inside and after a loop.
double around_loop(double x, int n) {
  if (n == -1)
    return x * x * x;
  double s = x;
  for (int i = 0; i < 3; ++i) {
    double d = s + 1. + i;
    if (i == n)
      return x / d;
    s += x / d;
  }
  double e = s * s + 1.;
  if (n == 10)
    return x / e;
  double f = e + x;
  return s / f;
}

bool close(double a, double b) {
  return std::isfinite(a) && std::fabs(a - b) < 1e-6 * (1. + std::fabs(b));
}

#define FIRST(F, LO, HI)                                                       \
  {                                                                            \
    auto g = clad::gradient(F, "x");                                           \
    bool ok = true;                                                            \
    for (int n = LO; n <= HI; ++n) {                                           \
      const double x = 1.3, h = 1e-6;                                          \
      double d = 0;                                                            \
      g.execute(x, n, &d);                                                     \
      ok = ok && close(d, (F(x + h, n) - F(x - h, n)) / (2 * h));              \
    }                                                                          \
    printf("%s %d\n", #F, ok);                                                 \
  }

#define SECOND(F, LO, HI)                                                      \
  {                                                                            \
    auto g = clad::gradient(F, "x");                                           \
    auto hess = clad::hessian(F, "x");                                         \
    bool ok = true;                                                            \
    for (int n = LO; n <= HI; ++n) {                                           \
      const double x = 1.3, h = 1e-5;                                          \
      double d2 = 0, gp = 0, gm = 0;                                           \
      hess.execute(x, n, &d2);                                                 \
      g.execute(x + h, n, &gp);                                                \
      g.execute(x - h, n, &gm);                                                \
      ok = ok && close(d2, (gp - gm) / (2 * h));                               \
    }                                                                          \
    printf("%s %d\n", #F, ok);                                                 \
  }

int main() {
  FIRST(in_for, 0, 5)         // CHECK-EXEC: in_for 1
  FIRST(last_in_body, 0, 5)   // CHECK-EXEC-NEXT: last_in_body 1
  FIRST(nested, 0, 7)         // CHECK-EXEC-NEXT: nested 1
  FIRST(two_returns, 0, 9)    // CHECK-EXEC-NEXT: two_returns 1
  FIRST(in_while, 0, 5)       // CHECK-EXEC-NEXT: in_while 1
  FIRST(in_range_for, 0, 5)   // CHECK-EXEC-NEXT: in_range_for 1
  FIRST(around_loop, -1, 11)  // CHECK-EXEC-NEXT: around_loop 1
  // The pullback of a pushforward reads values back from tapes that the
  // returned expression pushes to.
  SECOND(in_for, 0, 5)        // CHECK-EXEC-NEXT: in_for 1
  SECOND(last_in_body, 0, 5)  // CHECK-EXEC-NEXT: last_in_body 1
  SECOND(nested, 0, 7)        // CHECK-EXEC-NEXT: nested 1
  SECOND(two_returns, 0, 9)   // CHECK-EXEC-NEXT: two_returns 1
  SECOND(in_while, 0, 5)      // CHECK-EXEC-NEXT: in_while 1
  SECOND(around_loop, -1, 11) // CHECK-EXEC-NEXT: around_loop 1
}
