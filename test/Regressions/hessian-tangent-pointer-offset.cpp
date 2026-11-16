// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A pointer-valued conditional is an rvalue, but it has an adjoint: the
// adjoint storage of the branch taken.
//
// A pushforward guards the tangent of a pointer derived from a tangent
// pointer that may be null, `_d_cv = _d_cvi ? _d_cvi + k : nullptr`. When a
// hessian differentiates that pushforward in reverse, the pointer still needs
// an adjoint, or nothing read through it passes anything back: here the
// second derivatives through the array the callee reads at an offset.

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

const int K = 3;
const double data[2 * 2 * K] = {1.1, 0.4, -0.3, 0.9, 0.2, 0.7,
                          1.3, -0.5, 0.6, 0.8, 0.1, -0.2};

void fill(const double* cvi, const double* p, int c, int side, double L,
          double* I) {
  const double* cv = cvi + (c * 2 + side) * K;
  const double f = cv[0] * (1. + p[2]);
  I[0] = f;
  I[1] = cv[1] * L + f * L * L;
}

// CHECK: void fill_pushforward_pullback(
// CHECK: double *_d_d_cv = (_cond0 ? _d_d_cvi + (c * 2 + side) * 3 : nullptr);

double kernel(const double* p) {
  // the array the callee reads depends on the parameters
  double cvi[2 * 2 * K];
  for (int k = 0; k < 2 * 2 * K; ++k)
    cvi[k] = data[k];
  const double dln = std::log(p[1]);
  const double a = p[0] * 0.0795774715;
  for (int c = 0; c < 2; ++c)
    for (int sd = 0; sd < 2; ++sd) {
      const int off = (c * 2 + sd) * K;
      cvi[off] += dln * (a * data[off + 1] + a * a * data[off + 2]);
      cvi[off + 1] += dln * a * data[off + 2];
    }
  double total = 0.;
  for (int c = 0; c < 2; ++c) {
    double IA[2], IB[2];
    fill(cvi, p, c, 0, dln, IA);
    fill(cvi, p, c, 1, dln, IB);
    total += IA[0] * IB[0] + p[0] * (IA[1] * IB[0] + IA[0] * IB[1]);
  }
  return total;
}

// The same in a gradient, written out: the two branches lead to different
// storage, through one and two levels of pointers.
double pick(double x, int c) {
  double a = x, b = 2 * x;
  const double* r = c ? &a : &b;
  return r[0] * r[0];
}

double pick2(double x, int c) {
  double a = 0, b = 0;
  double* p = &a;
  double* q = &b;
  double** r = c ? &p : &q;
  **r = x * x;
  return 3 * a + 5 * b;
}

int main() {
  auto dpick = clad::gradient(pick, "x");
  auto dpick2 = clad::gradient(pick2, "x");
  for (int c = 0; c < 2; ++c) {
    double d1 = 0, d2 = 0;
    dpick.execute(1.5, c, &d1);
    dpick2.execute(1.5, c, &d2);
    printf("%.2f %.2f\n", d1, d2);
  }
  // CHECK-EXEC: 12.00 15.00
  // CHECK-EXEC-NEXT: 3.00 9.00

  auto grad = clad::gradient(kernel, "p");
  auto hess = clad::hessian(kernel, "p[0:2]");
  double p[3] = {0.118, 1.3, 0.2}, H[9] = {};
  hess.execute(p, H);
  // every entry against central differences of the exact gradient
  double worst = 0., scale = 0.;
  for (int i = 0; i < 3; ++i) {
    const double h = 1e-6 * p[i];
    double pp[3] = {p[0], p[1], p[2]}, pm[3] = {p[0], p[1], p[2]};
    double gp[3] = {}, gm[3] = {};
    pp[i] += h;
    pm[i] -= h;
    grad.execute(pp, gp);
    grad.execute(pm, gm);
    for (int j = 0; j < 3; ++j) {
      const double fd = (gp[j] - gm[j]) / (2 * h);
      scale = std::fmax(scale, std::fabs(fd));
      worst = std::fmax(worst, std::fabs(H[j * 3 + i] - fd));
    }
  }
  printf("%.6f %.6f %.6f\n", H[0], H[1], H[2]); // CHECK-EXEC-NEXT: 0.016663 2.684442 0.769819
  printf("%.6f %.6f %.6f\n", H[3], H[4], H[5]); // CHECK-EXEC-NEXT: 2.684442 0.575448 0.498350
  printf("%.6f %.6f %.6f\n", H[6], H[7], H[8]); // CHECK-EXEC-NEXT: 0.769819 0.498350 4.127541
  printf("%s\n", worst <= 1e-6 * scale ? "agrees with finite differences"
                                        : "DIFFERS from finite differences");
  // CHECK-EXEC-NEXT: agrees with finite differences
}
