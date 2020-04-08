#ifndef ALGOIM_STENCILPOLY_HPP
#define ALGOIM_STENCILPOLY_HPP

/* Algoim::StencilPoly<N,Degree> implements stencil-based polynomial interpolation algorithms
   for 10 different classes of polynomial in 2D and 3D on uniform Cartesian grids. In particular,
   the templated typedef allows the (N,Degree) parameter pair to be translated to the specific
   polynomial class implementing the interpolation algorithm:
       N = 2 or 3.
       Degree = -1: choose the bicubic/tricubic interpolant.
       Degree \in {2,3,4,5}: choose the corresponding "Taylor" polynomial stencil
   If (N,Degree) is undefined, a compiler error relating to "InvalidPoly" shall be generated.

   For more information, refer to the paper
       R. I. Saye, High-order methods for computing distances to implicitly defined surfaces,
        Communications in Applied Mathematics and Computational Science, 9(1), 107-141 (2014),
        http://dx.doi.org/10.2140/camcos.2014.9.107
*/

#include <vector>
#include "algoim_blitzinc.hpp"
#include "algoim_utility.hpp"
#include "algoim_stencilpoly_detail.hpp"

namespace Algoim
{
    namespace detail
    {
        // Calculate the coefficients for a polynomial in N dimensions, based on a given stencil of O points
        // evaluated on a rectangular grid, where M is the number of coefficients in the polynomial
        template<int N, int M, int O, typename F>
        void calculateCoefficients(TinyVector<double,M>& c, const TinyVector<int,N>* stencil, const double* matrix, const TinyVector<int,N>& gp, const F& phi)
        {
            TinyVector<double,O> values;
            for (int i = 0; i < O; ++i)
                values(i) = phi(TinyVector<int,N>(gp + stencil[i]));
            for (int i = 0; i < M; ++i)
            {
                c(i) = 0.0;
                for (int j = 0; j < O; ++j)
                    c(i) += matrix[i*O + j]*values(j);
            }
        }

        /* Two-dimensional, degree 2 polynomial of the form
                p(x,y) = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2
            based on a 12-point stencil. Pseudoinverse has rank 6. */
        struct N2_PolyDegree2
        {
            enum { order = 3 };

            TinyVector<double,6> c;

            N2_PolyDegree2() {}

            template<typename F>
            N2_PolyDegree2(const TinyVector<int,2>& i, const F& phi, const TinyVector<double,2>& dx)
            {
                detail::calculateCoefficients<2,6,12>(c, detail::StencilPolyData::N2_stencil12points(), detail::StencilPolyData::N2_polyDegree2Inverse(), i, phi);
                TinyVector<double,2> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(0)*dxinv(0);
                c(4) *= dxinv(0)*dxinv(1);
                c(5) *= dxinv(1)*dxinv(1);
            }

            double operator() (const TinyVector<double,2>& x) const
            {
                return c(0) + x(1)*(c(2) + x(1)*c(5)) + x(0)*(c(1) + x(1)*c(4) + x(0)*c(3));
            }

            TinyVector<double,2> grad(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,2>(c(1) + x(1)*c(4) + x(0)*(2.0*c(3)), c(2) + x(1)*(2.0*c(5)) + x(0)*c(4));
            }

            TinyVector<double,3> hessian(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,3>(2.0*c(3), c(4), 2.0*c(5));
            }
        };

        /* Two-dimensional, degree 3 polynomial of the form
                p(x,y) = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2 + c6 x^3 + c7 x^2 y + c8 x y^2 + c9 y^3
            based on a 12-point stencil. Pseudoinverse has rank 10.*/
        struct N2_PolyDegree3
        {
            enum { order = 4 };

            TinyVector<double,10> c;

            N2_PolyDegree3() {}

            template<typename F>
            N2_PolyDegree3(const TinyVector<int,2>& i, const F& phi, const TinyVector<double,2>& dx)
            {
                detail::calculateCoefficients<2,10,12>(c, detail::StencilPolyData::N2_stencil12points(), detail::StencilPolyData::N2_polyDegree3Inverse(), i, phi);
                TinyVector<double,2> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(0)*dxinv(0);
                c(4) *= dxinv(0)*dxinv(1);
                c(5) *= dxinv(1)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(0)*dxinv(0);
                c(7) *= dxinv(0)*dxinv(0)*dxinv(1);
                c(8) *= dxinv(0)*dxinv(1)*dxinv(1);
                c(9) *= dxinv(1)*dxinv(1)*dxinv(1);
            }

            double operator() (const TinyVector<double,2>& x) const
            {
                return c(0) + x(1)*(c(2) + x(1)*(c(5) + x(1)*c(9))) + x(0)*(c(1) + x(1)*(c(4) + x(1)*c(8)) + x(0)*(c(3) + x(1)*c(7) + x(0)*c(6)));
            }

            TinyVector<double,2> grad(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,2>(c(1) + x(1)*(c(4) + x(1)*c(8)) + x(0)*(2.0*c(3) + x(1)*(2.0*c(7)) + x(0)*(3.0*c(6))),
                    c(2) + x(1)*(2.0*c(5) + x(1)*(3.0*c(9))) + x(0)*(c(4) + x(1)*(2.0*c(8)) + x(0)*c(7)));
            }

            TinyVector<double,3> hessian(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,3>(2.0*c(3) + x(1)*(2.0*c(7)) + x(0)*(6.0*c(6)), c(4) + x(1)*(2.0*c(8)) + x(0)*(2.0*c(7)), 2.0*c(5) + x(1)*(6.0*c(9)) + x(0)*(2.0*c(8)));
            }
        };

        /* Two-dimensional, degree 4 polynomial of the form 
                p(x,y) = c0 + c1 x + c2 y + c3 x^2 + c4 x y + c5 y^2 + c6 x^3 + c7 x^2 y + c8 x y^2 + c9 y^3 +
                        c10 x^4 + c11 x^3 y + c12 x^2 y^2 + c13 x y^3 + c14 y^4
            based on a 24-point stencil. Pseudoinverse has rank 15. (Note that for a stencil based on a 4 by 4
            patch, as used in bicubic, the Vandermonde matrix has rank 13.) */
        struct N2_PolyDegree4
        {
            enum { order = 5 };

            TinyVector<double,15> c;

            N2_PolyDegree4() {};

            template<typename F>
            N2_PolyDegree4(const TinyVector<int,2>& i, const F& phi, const TinyVector<double,2>& dx)
            {
                detail::calculateCoefficients<2,15,24>(c, detail::StencilPolyData::N2_stencil24points(), detail::StencilPolyData::N2_polyDegree4Inverse(), i, phi);
                TinyVector<double,2> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(0)*dxinv(0);
                c(4) *= dxinv(0)*dxinv(1);
                c(5) *= dxinv(1)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(0)*dxinv(0);
                c(7) *= dxinv(0)*dxinv(0)*dxinv(1);
                c(8) *= dxinv(0)*dxinv(1)*dxinv(1);
                c(9) *= dxinv(1)*dxinv(1)*dxinv(1);
                c(10) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0);
                c(11) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1);
                c(12) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1);
                c(13) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1);
                c(14) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
            }

            double operator() (const TinyVector<double,2>& x) const
            {
                return c(0) + x(1)*(c(2) + x(1)*(c(5) + x(1)*(c(9) + x(1)*c(14)))) + x(0)*(c(1) + x(1)*(c(4) + x(1)*(c(8) + x(1)*c(13))) + x(0)*(c(3) + x(1)*(c(7) + x(1)*c(12)) + x(0)*(c(6) + x(1)*c(11) + x(0)*c(10))));
            }

            TinyVector<double,2> grad(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,2>(c(1) + x(1)*(c(4) + x(1)*(c(8) + x(1)*c(13))) + x(0)*(2.0*c(3) + x(1)*(2.0*c(7) + x(1)*(2.0*c(12))) + x(0)*(3.0*c(6) + x(1)*(3.0*c(11)) + x(0)*(4.0*c(10)))),
                    c(2) + x(1)*(2.0*c(5) + x(1)*(3.0*c(9) + x(1)*(4.0*c(14)))) + x(0)*(c(4) + x(1)*(2.0*c(8) + x(1)*(3.0*c(13))) + x(0)*(c(7) + x(1)*(2.0*c(12)) + x(0)*c(11))));
            }

            TinyVector<double,3> hessian(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,3>(2.0*c(3) + x(1)*(2.0*c(7) + x(1)*(2.0*c(12))) + x(0)*(6.0*c(6) + x(1)*(6.0*c(11)) + x(0)*(12.0*c(10))),
                    c(4) + x(1)*(2.0*c(8) + x(1)*(3.0*c(13))) + x(0)*(2.0*c(7) + x(1)*(4.0*c(12)) + x(0)*(3.0*c(11))),
                    2.0*c(5) + x(1)*(6.0*c(9) + x(1)*(12.0*c(14))) + x(0)*(2.0*c(8) + x(1)*(6.0*c(13)) + x(0)*(2.0*c(12))));
            }
        };

        /* Two-dimensional, degree 5 polynomial of the form 
                p(x,y) = c0 + c1 x + c2 y + ... + c20 y^5
            based on a 24-point stencil. Pseudoinverse has rank 21.*/
        struct N2_PolyDegree5
        {
            enum { order = 6 };

            TinyVector<double,21> c;

            N2_PolyDegree5() {};

            template<typename F>
            N2_PolyDegree5(const TinyVector<int,2>& i, const F& phi, const TinyVector<double,2>& dx)
            {
                detail::calculateCoefficients<2,21,24>(c, detail::StencilPolyData::N2_stencil24points(), detail::StencilPolyData::N2_polyDegree5Inverse(), i, phi);
                TinyVector<double,2> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(0)*dxinv(0);
                c(4) *= dxinv(0)*dxinv(1);
                c(5) *= dxinv(1)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(0)*dxinv(0);
                c(7) *= dxinv(0)*dxinv(0)*dxinv(1);
                c(8) *= dxinv(0)*dxinv(1)*dxinv(1);
                c(9) *= dxinv(1)*dxinv(1)*dxinv(1);
                c(10) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0);
                c(11) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1);
                c(12) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1);
                c(13) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1);
                c(14) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
                c(15) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0);
                c(16) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1);
                c(17) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1);
                c(18) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1);
                c(19) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
                c(20) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
            }

            double operator() (const TinyVector<double,2>& x) const
            {
                return c(0) + x(1)*(c(2) + x(1)*(c(5) + x(1)*(c(9) + x(1)*(c(14) + x(1)*c(20))))) + x(0)*(c(1) + x(1)*(c(4) + x(1)*(c(8) + x(1)*(c(13) + x(1)*c(19)))) + x(0)*(c(3) + x(1)*(c(7) + x(1)*(c(12) + x(1)*c(18))) + x(0)*(c(6) + x(1)*(c(11) + x(1)*c(17)) + x(0)*(c(10) + x(1)*c(16) + x(0)*c(15)))));
            }

            TinyVector<double,2> grad(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,2>(c(1) + x(1)*(c(4) + x(1)*(c(8) + x(1)*(c(13) + x(1)*c(19)))) + x(0)*(2.0*c(3) + x(1)*(2.0*c(7) + x(1)*(2.0*c(12) + x(1)*(2.0*c(18)))) + x(0)*(3.0*c(6) + x(1)*(3.0*c(11) + x(1)*(3.0*c(17))) + x(0)*(4.0*c(10) + x(1)*(4.0*c(16)) + x(0)*(5.0*c(15))))),
                    c(2) + x(1)*(2.0*c(5) + x(1)*(3.0*c(9) + x(1)*(4.0*c(14) + x(1)*(5.0*c(20))))) + x(0)*(c(4) + x(1)*(2.0*c(8) + x(1)*(3.0*c(13) + x(1)*(4.0*c(19)))) + x(0)*(c(7) + x(1)*(2.0*c(12) + x(1)*(3.0*c(18))) + x(0)*(c(11) + x(1)*(2.0*c(17)) + x(0)*c(16)))));
            }

            TinyVector<double,3> hessian(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,3>(2.0*c(3) + x(1)*(2.0*c(7) + x(1)*(2.0*c(12) + x(1)*(2.0*c(18)))) + x(0)*(6.0*c(6) + x(1)*(6.0*c(11) + x(1)*(6.0*c(17))) + x(0)*(12.0*c(10) + x(1)*(12.0*c(16)) + x(0)*(20.0*c(15)))),
                    c(4) + x(1)*(2.0*c(8) + x(1)*(3.0*c(13) + x(1)*(4.0*c(19)))) + x(0)*(2.0*c(7) + x(1)*(4.0*c(12) + x(1)*(6.0*c(18))) + x(0)*(3.0*c(11) + x(1)*(6.0*c(17)) + x(0)*(4.0*c(16)))),
                    2.0*c(5) + x(1)*(6.0*c(9) + x(1)*(12.0*c(14) + x(1)*(20.0*c(20)))) + x(0)*(2.0*c(8) + x(1)*(6.0*c(13) + x(1)*(12.0*c(19))) + x(0)*(2.0*c(12) + x(1)*(6.0*c(18)) + x(0)*(2.0*c(17)))));
            }
        };

        /* Two-dimensional, bicubic interpolant guaranteeing C^1 continuity between cells on a rectangular
            grid, using second order finite differences evaluated at gird points to ensure continuity of
            the gradient across grid cells. This interpolant is equivalent to that discussed in D. L. Chopp, 
            Some improvements of the Fast Marching Method, SIAM Journal Scientific Computing 23(1) (2001) 230–244. */
        struct N2_Bicubic
        {
            enum { order = 3 };

            TinyVector<double,16> c;

            template<typename F>
            N2_Bicubic(const TinyVector<int,2>& i, const F& phi, const TinyVector<double,2>& dx)
            {
                TinyVector<int,2> stencilext = 4;
                blitz::Array<double,2> stencil(stencilext);

                for (MultiLoop<2> b(0, 4); b; ++b)
                {
                    TinyVector<int,2> j = i - 1 + b();
                    stencil(b()) = phi(j);
                }

                const double* rhs = stencil.data();
                for (int i = 0; i < 16; ++i)
                {
                    c(i) = 0.0;
                    for (int j = 0; j < 16; ++j)
                        c(i) += detail::StencilPolyData::N2_bicubicInverse()[i*16+j]*rhs[j];
                }

                TinyVector<double,2> dxinv = 1.0 / dx;
                double sx = 1.0;
                for (int i = 0; i < 4; ++i)
                {
                    double sy = 1.0;
                    for (int j = 0; j < 4; ++j)
                    {
                        c(i*4+j) *= sx*sy;
                        sy *= dxinv(1);
                    }
                    sx *= dxinv(0);
                }
            }

            double operator() (const TinyVector<double,2>& x) const
            {
                return c(0) + x(1)*(c(1) + x(1)*(c(2) + x(1)*c(3))) + x(0)*(c(4) + x(1)*(c(5) + x(1)*(c(6) + x(1)*c(7))) + x(0)*(c(8) + x(1)*(c(9) + x(1)*(c(10) + x(1)*c(11))) + x(0)*(c(12) + x(1)*(c(13) + x(1)*(c(14) + x(1)*c(15))))));
            }

            TinyVector<double,2> grad(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,2>(c(4) + x(1)*(c(5) + x(1)*(c(6) + x(1)*c(7))) + x(0)*(2.0*c(8) + x(1)*(2.0*c(9) + x(1)*(2.0*c(10) + x(1)*(2.0*c(11)))) + x(0)*(3.0*c(12) + x(1)*(3.0*c(13) + x(1)*(3.0*c(14) + x(1)*(3.0*c(15)))))),
                    c(1) + x(1)*(2.0*c(2) + x(1)*(3.0*c(3))) + x(0)*(c(5) + x(1)*(2.0*c(6) + x(1)*(3.0*c(7))) + x(0)*(c(9) + x(1)*(2.0*c(10) + x(1)*(3.0*c(11))) + x(0)*(c(13) + x(1)*(2.0*c(14) + x(1)*(3.0*c(15)))))));
            }

            TinyVector<double,3> hessian(const TinyVector<double,2>& x) const
            {
                return TinyVector<double,3>(2.0*c(8) + x(1)*(2.0*c(9) + x(1)*(2.0*c(10) + x(1)*(2.0*c(11)))) + x(0)*(6.0*c(12) + x(1)*(6.0*c(13) + x(1)*(6.0*c(14) + x(1)*(6.0*c(15))))),
                    c(5) + x(1)*(2.0*c(6) + x(1)*(3.0*c(7))) + x(0)*(2.0*c(9) + x(1)*(4.0*c(10) + x(1)*(6.0*c(11))) + x(0)*(3.0*c(13) + x(1)*(6.0*c(14) + x(1)*(9.0*c(15))))),
                    2.0*c(2) + x(1)*(6.0*c(3)) + x(0)*(2.0*c(6) + x(1)*(6.0*c(7)) + x(0)*(2.0*c(10) + x(1)*(6.0*c(11)) + x(0)*(2.0*c(14) + x(1)*(6.0*c(15))))));
            }
        };

        /* Three-dimensional, degree 2 polynomial of the form
                p(x,y) = c0 + c1 x + c2 y + c3 z + c4 x^2 + c5 x y + c6 x z + c7 y^2 + c8 y z + c9 z^2
            based on a 32-point stencil. Pseudoinverse has rank 10. */
        struct N3_PolyDegree2
        {
            enum { order = 3 };

            TinyVector<double,10> c;

            N3_PolyDegree2() {}

            template<typename F>
            N3_PolyDegree2(const TinyVector<int,3>& i, const F& phi, const TinyVector<double,3>& dx)
            {
                detail::calculateCoefficients<3,10,32>(c, detail::StencilPolyData::N3_stencil32points(), detail::StencilPolyData::N3_polyDegree2Inverse(), i, phi);
                TinyVector<double,3> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(2);
                c(4) *= dxinv(0)*dxinv(0);
                c(5) *= dxinv(0)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(2);
                c(7) *= dxinv(1)*dxinv(1);
                c(8) *= dxinv(1)*dxinv(2);
                c(9) *= dxinv(2)*dxinv(2);
            }

            double operator() (const TinyVector<double,3>& x) const
            {
                return c(0) + x(2)*(c(3) + x(2)*c(9)) + x(1)*(c(2) + x(2)*c(8) + x(1)*c(7)) + x(0)*(c(1) + x(2)*c(6) + x(1)*c(5) + x(0)*c(4));
            }

            TinyVector<double,3> grad(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,3>(c(1) + x(2)*c(6) + x(1)*c(5) + x(0)*(2.0*c(4)), c(2) + x(2)*c(8) + x(1)*(2.0*c(7)) + x(0)*c(5), c(3) + x(2)*(2.0*c(9)) + x(1)*c(8) + x(0)*c(6));
            }

            TinyVector<double,6> hessian(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,6>(2.0*c(4), c(5), c(6), 2.0*c(7), c(8), 2.0*c(9));
            }
        };

        /* Three-dimensional, degree 3 polynomial of the form
                p(x,y) = c0 + c1 x + c2 y + c3 z + c4 x^2 + c5 x y + c6 x z + c7 y^2 + c8 y z + c9 z^2 +
                        c10 x^3 + c11 x^2 y + c12 x^2 z + c13 x y^2 + c14 x y z + c15 x z^2 + c16 y^3 + c17 y^2 z + c18 y z^2 + c19 z^3
            based on a 32-point stencil. Pseudoinverse has rank 20. */
        struct N3_PolyDegree3
        {
            enum { order = 4 };

            TinyVector<double,20> c;

            N3_PolyDegree3() {}

            template<typename F>
            N3_PolyDegree3(const TinyVector<int,3>& i, const F& phi, const TinyVector<double,3>& dx)
            {
                detail::calculateCoefficients<3,20,32>(c, detail::StencilPolyData::N3_stencil32points(), detail::StencilPolyData::N3_polyDegree3Inverse(), i, phi);
                TinyVector<double,3> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(2);
                c(4) *= dxinv(0)*dxinv(0);
                c(5) *= dxinv(0)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(2);
                c(7) *= dxinv(1)*dxinv(1);
                c(8) *= dxinv(1)*dxinv(2);
                c(9) *= dxinv(2)*dxinv(2);
                c(10) *= dxinv(0)*dxinv(0)*dxinv(0);
                c(11) *= dxinv(0)*dxinv(0)*dxinv(1);
                c(12) *= dxinv(0)*dxinv(0)*dxinv(2);
                c(13) *= dxinv(0)*dxinv(1)*dxinv(1);
                c(14) *= dxinv(0)*dxinv(1)*dxinv(2);
                c(15) *= dxinv(0)*dxinv(2)*dxinv(2);
                c(16) *= dxinv(1)*dxinv(1)*dxinv(1);
                c(17) *= dxinv(1)*dxinv(1)*dxinv(2);
                c(18) *= dxinv(1)*dxinv(2)*dxinv(2);
                c(19) *= dxinv(2)*dxinv(2)*dxinv(2);
            }

            double operator() (const TinyVector<double,3>& x) const
            {
                return c(0) + x(2)*(c(3) + x(2)*(c(9) + x(2)*c(19))) + x(1)*(c(2) + x(2)*(c(8) + x(2)*c(18)) + x(1)*(c(7) + x(2)*c(17) + x(1)*c(16))) + x(0)*(c(1) + x(2)*(c(6) + x(2)*c(15)) + x(1)*(c(5) + x(2)*c(14) + x(1)*c(13)) + x(0)*(c(4) + x(2)*c(12) + x(1)*c(11) + x(0)*c(10)));
            }

            TinyVector<double,3> grad(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,3>(c(1) + x(2)*(c(6) + x(2)*c(15)) + x(1)*(c(5) + x(2)*c(14) + x(1)*c(13)) + x(0)*(2.0*c(4) + x(2)*(2.0*c(12)) + x(1)*(2.0*c(11)) + x(0)*(3.0*c(10))),
                    c(2) + x(2)*(c(8) + x(2)*c(18)) + x(1)*(2.0*c(7) + x(2)*(2.0*c(17)) + x(1)*(3.0*c(16))) + x(0)*(c(5) + x(2)*c(14) + x(1)*(2.0*c(13)) + x(0)*c(11)),
                    c(3) + x(2)*(2.0*c(9) + x(2)*(3.0*c(19))) + x(1)*(c(8) + x(2)*(2.0*c(18)) + x(1)*c(17)) + x(0)*(c(6) + x(2)*(2.0*c(15)) + x(1)*c(14) + x(0)*c(12)));
            }

            TinyVector<double,6> hessian(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,6>(2.0*c(4) + x(2)*(2.0*c(12)) + x(1)*(2.0*c(11)) + x(0)*(6.0*c(10)), c(5) + x(2)*c(14) + x(1)*(2.0*c(13)) + x(0)*(2.0*c(11)),
                    c(6) + x(2)*(2.0*c(15)) + x(1)*c(14) + x(0)*(2.0*c(12)), 2.0*c(7) + x(2)*(2.0*c(17)) + x(1)*(6.0*c(16)) + x(0)*(2.0*c(13)),
                    c(8) + x(2)*(2.0*c(18)) + x(1)*(2.0*c(17)) + x(0)*c(14), 2.0*c(9) + x(2)*(6.0*c(19)) + x(1)*(2.0*c(18)) + x(0)*(2.0*c(15)));
            }
        };

        /* Three-dimensional, degree 4 polynomial of the form
                p(x,y) = c0 + c1 x + c2 y + c3 z + c4 x^2 + c5 x y + c6 x z + c7 y^2 + c8 y z + c9 z^2 +
                        c10 x^3 + c11 x^2 y + c12 x^2 z + c13 x y^2 + c14 x y z + c15 x z^2 + c16 y^3 + c17 y^2 z + c18 y z^2 + c19 z^3 +
                        c20 x^4 + c21 x^3 y + c22 x^3 z + c23 x^2 y^2 + c24 x^2 y z + c25 x^2 z^2 + c26 x y^3 + c27 x y^2 z + c28 x y z^2 +
                        c29 x z^3 + c30 y^4 + c31 y^3 z + c32 y^2 z^2 + c33 y z^3 + c34 z^4
            based on an 88-point stencil. Pseudoinverse has rank 35. */
        struct N3_PolyDegree4
        {
            enum { order = 5 };

            TinyVector<double,35> c;

            N3_PolyDegree4() {}

            template<typename F>
            N3_PolyDegree4(const TinyVector<int,3>& i, const F& phi, const TinyVector<double,3>& dx)
            {
                detail::calculateCoefficients<3,35,88>(c, detail::StencilPolyData::N3_stencil88points(), detail::StencilPolyData::N3_polyDegree4Inverse(), i, phi);
                TinyVector<double,3> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(2);
                c(4) *= dxinv(0)*dxinv(0);
                c(5) *= dxinv(0)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(2);
                c(7) *= dxinv(1)*dxinv(1);
                c(8) *= dxinv(1)*dxinv(2);
                c(9) *= dxinv(2)*dxinv(2);
                c(10) *= dxinv(0)*dxinv(0)*dxinv(0);
                c(11) *= dxinv(0)*dxinv(0)*dxinv(1);
                c(12) *= dxinv(0)*dxinv(0)*dxinv(2);
                c(13) *= dxinv(0)*dxinv(1)*dxinv(1);
                c(14) *= dxinv(0)*dxinv(1)*dxinv(2);
                c(15) *= dxinv(0)*dxinv(2)*dxinv(2);
                c(16) *= dxinv(1)*dxinv(1)*dxinv(1);
                c(17) *= dxinv(1)*dxinv(1)*dxinv(2);
                c(18) *= dxinv(1)*dxinv(2)*dxinv(2);
                c(19) *= dxinv(2)*dxinv(2)*dxinv(2);
                c(20) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0);
                c(21) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1);
                c(22) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(2);
                c(23) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1);
                c(24) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(2);
                c(25) *= dxinv(0)*dxinv(0)*dxinv(2)*dxinv(2);
                c(26) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1);
                c(27) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(2);
                c(28) *= dxinv(0)*dxinv(1)*dxinv(2)*dxinv(2);
                c(29) *= dxinv(0)*dxinv(2)*dxinv(2)*dxinv(2);
                c(30) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
                c(31) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(2);
                c(32) *= dxinv(1)*dxinv(1)*dxinv(2)*dxinv(2);
                c(33) *= dxinv(1)*dxinv(2)*dxinv(2)*dxinv(2);
                c(34) *= dxinv(2)*dxinv(2)*dxinv(2)*dxinv(2);
            }

            double operator() (const TinyVector<double,3>& x) const
            {
                return c(0) + x(2)*(c(3) + x(2)*(c(9) + x(2)*(c(19) + x(2)*c(34)))) + x(1)*(c(2) + x(2)*(c(8) + x(2)*(c(18) + x(2)*c(33))) + x(1)*(c(7) + x(2)*(c(17) + x(2)*c(32)) + x(1)*(c(16) + x(2)*c(31) + x(1)*c(30)))) + x(0)*(c(1) + x(2)*(c(6) + x(2)*(c(15) + x(2)*c(29))) + x(1)*(c(5) + x(2)*(c(14) + x(2)*c(28)) + x(1)*(c(13) + x(2)*c(27) + x(1)*c(26))) + x(0)*(c(4) + x(2)*(c(12) + x(2)*c(25)) + x(1)*(c(11) + x(2)*c(24) + x(1)*c(23)) + x(0)*(c(10) + x(2)*c(22) + x(1)*c(21) + x(0)*c(20))));
            }

            TinyVector<double,3> grad(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,3>(c(1) + x(2)*(c(6) + x(2)*(c(15) + x(2)*c(29))) + x(1)*(c(5) + x(2)*(c(14) + x(2)*c(28)) + x(1)*(c(13) + x(2)*c(27) + x(1)*c(26))) + x(0)*(2.0*c(4) + x(2)*(2.0*c(12) + x(2)*(2.0*c(25))) + x(1)*(2.0*c(11) + x(2)*(2.0*c(24)) + x(1)*(2.0*c(23))) + x(0)*(3.0*c(10) + x(2)*(3.0*c(22)) + x(1)*(3.0*c(21)) + x(0)*(4.0*c(20)))),
                    c(2) + x(2)*(c(8) + x(2)*(c(18) + x(2)*c(33))) + x(1)*(2.0*c(7) + x(2)*(2.0*c(17) + x(2)*(2.0*c(32))) + x(1)*(3.0*c(16) + x(2)*(3.0*c(31)) + x(1)*(4.0*c(30)))) + x(0)*(c(5) + x(2)*(c(14) + x(2)*c(28)) + x(1)*(2.0*c(13) + x(2)*(2.0*c(27)) + x(1)*(3.0*c(26))) + x(0)*(c(11) + x(2)*c(24) + x(1)*(2.0*c(23)) + x(0)*c(21))),
                    c(3) + x(2)*(2.0*c(9) + x(2)*(3.0*c(19) + x(2)*(4.0*c(34)))) + x(1)*(c(8) + x(2)*(2.0*c(18) + x(2)*(3.0*c(33))) + x(1)*(c(17) + x(2)*(2.0*c(32)) + x(1)*c(31))) + x(0)*(c(6) + x(2)*(2.0*c(15) + x(2)*(3.0*c(29))) + x(1)*(c(14) + x(2)*(2.0*c(28)) + x(1)*c(27)) + x(0)*(c(12) + x(2)*(2.0*c(25)) + x(1)*c(24) + x(0)*c(22))));
            }

            TinyVector<double,6> hessian(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,6>(2.0*c(4) + x(2)*(2.0*c(12) + x(2)*(2.0*c(25))) + x(1)*(2.0*c(11) + x(2)*(2.0*c(24)) + x(1)*(2.0*c(23))) + x(0)*(6.0*c(10) + x(2)*(6.0*c(22)) + x(1)*(6.0*c(21)) + x(0)*(12.0*c(20))),
                    c(5) + x(2)*(c(14) + x(2)*c(28)) + x(1)*(2.0*c(13) + x(2)*(2.0*c(27)) + x(1)*(3.0*c(26))) + x(0)*(2.0*c(11) + x(2)*(2.0*c(24)) + x(1)*(4.0*c(23)) + x(0)*(3.0*c(21))),
                    c(6) + x(2)*(2.0*c(15) + x(2)*(3.0*c(29))) + x(1)*(c(14) + x(2)*(2.0*c(28)) + x(1)*c(27)) + x(0)*(2.0*c(12) + x(2)*(4.0*c(25)) + x(1)*(2.0*c(24)) + x(0)*(3.0*c(22))),
                    2.0*c(7) + x(2)*(2.0*c(17) + x(2)*(2.0*c(32))) + x(1)*(6.0*c(16) + x(2)*(6.0*c(31)) + x(1)*(12.0*c(30))) + x(0)*(2.0*c(13) + x(2)*(2.0*c(27)) + x(1)*(6.0*c(26)) + x(0)*(2.0*c(23))),
                    c(8) + x(2)*(2.0*c(18) + x(2)*(3.0*c(33))) + x(1)*(2.0*c(17) + x(2)*(4.0*c(32)) + x(1)*(3.0*c(31))) + x(0)*(c(14) + x(2)*(2.0*c(28)) + x(1)*(2.0*c(27)) + x(0)*c(24)),
                    2.0*c(9) + x(2)*(6.0*c(19) + x(2)*(12.0*c(34))) + x(1)*(2.0*c(18) + x(2)*(6.0*c(33)) + x(1)*(2.0*c(32))) + x(0)*(2.0*c(15) + x(2)*(6.0*c(29)) + x(1)*(2.0*c(28)) + x(0)*(2.0*c(25))));
            }
        };

        /* Three-dimensional, degree 5 polynomial of the form
                p(x,y) = c0 + c1 x + c2 y + c3 z + ... + c55 z^5
            based on an 88-point stencil. Pseudoinverse has rank 56. */
        struct N3_PolyDegree5
        {
            enum { order = 6 };

            TinyVector<double,56> c;

            N3_PolyDegree5() {}

            template<typename F>
            N3_PolyDegree5(const TinyVector<int,3>& i, const F& phi, const TinyVector<double,3>& dx)
            {
                detail::calculateCoefficients<3,56,88>(c, detail::StencilPolyData::N3_stencil88points(), detail::StencilPolyData::N3_polyDegree5Inverse(), i, phi);
                TinyVector<double,3> dxinv = 1.0 / dx;
                c(1) *= dxinv(0);
                c(2) *= dxinv(1);
                c(3) *= dxinv(2);
                c(4) *= dxinv(0)*dxinv(0);
                c(5) *= dxinv(0)*dxinv(1);
                c(6) *= dxinv(0)*dxinv(2);
                c(7) *= dxinv(1)*dxinv(1);
                c(8) *= dxinv(1)*dxinv(2);
                c(9) *= dxinv(2)*dxinv(2);
                c(10) *= dxinv(0)*dxinv(0)*dxinv(0);
                c(11) *= dxinv(0)*dxinv(0)*dxinv(1);
                c(12) *= dxinv(0)*dxinv(0)*dxinv(2);
                c(13) *= dxinv(0)*dxinv(1)*dxinv(1);
                c(14) *= dxinv(0)*dxinv(1)*dxinv(2);
                c(15) *= dxinv(0)*dxinv(2)*dxinv(2);
                c(16) *= dxinv(1)*dxinv(1)*dxinv(1);
                c(17) *= dxinv(1)*dxinv(1)*dxinv(2);
                c(18) *= dxinv(1)*dxinv(2)*dxinv(2);
                c(19) *= dxinv(2)*dxinv(2)*dxinv(2);
                c(20) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0);
                c(21) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1);
                c(22) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(2);
                c(23) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1);
                c(24) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(2);
                c(25) *= dxinv(0)*dxinv(0)*dxinv(2)*dxinv(2);
                c(26) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1);
                c(27) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(2);
                c(28) *= dxinv(0)*dxinv(1)*dxinv(2)*dxinv(2);
                c(29) *= dxinv(0)*dxinv(2)*dxinv(2)*dxinv(2);
                c(30) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
                c(31) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(2);
                c(32) *= dxinv(1)*dxinv(1)*dxinv(2)*dxinv(2);
                c(33) *= dxinv(1)*dxinv(2)*dxinv(2)*dxinv(2);
                c(34) *= dxinv(2)*dxinv(2)*dxinv(2)*dxinv(2);
                c(35) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0);
                c(36) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1);
                c(37) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(0)*dxinv(2);
                c(38) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1);
                c(39) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(1)*dxinv(2);
                c(40) *= dxinv(0)*dxinv(0)*dxinv(0)*dxinv(2)*dxinv(2);
                c(41) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1);
                c(42) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(1)*dxinv(2);
                c(43) *= dxinv(0)*dxinv(0)*dxinv(1)*dxinv(2)*dxinv(2);
                c(44) *= dxinv(0)*dxinv(0)*dxinv(2)*dxinv(2)*dxinv(2);
                c(45) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
                c(46) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(1)*dxinv(2);
                c(47) *= dxinv(0)*dxinv(1)*dxinv(1)*dxinv(2)*dxinv(2);
                c(48) *= dxinv(0)*dxinv(1)*dxinv(2)*dxinv(2)*dxinv(2);
                c(49) *= dxinv(0)*dxinv(2)*dxinv(2)*dxinv(2)*dxinv(2);
                c(50) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1);
                c(51) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(1)*dxinv(2);
                c(52) *= dxinv(1)*dxinv(1)*dxinv(1)*dxinv(2)*dxinv(2);
                c(53) *= dxinv(1)*dxinv(1)*dxinv(2)*dxinv(2)*dxinv(2);
                c(54) *= dxinv(1)*dxinv(2)*dxinv(2)*dxinv(2)*dxinv(2);
                c(55) *= dxinv(2)*dxinv(2)*dxinv(2)*dxinv(2)*dxinv(2);
            }

            double operator() (const TinyVector<double,3>& x) const
            {
                return c(0) + x(2)*(c(3) + x(2)*(c(9) + x(2)*(c(19) + x(2)*(c(34) + x(2)*c(55))))) + x(1)*(c(2) + x(2)*(c(8) + x(2)*(c(18) + x(2)*(c(33) + x(2)*c(54)))) + x(1)*(c(7) + x(2)*(c(17) + x(2)*(c(32) + x(2)*c(53))) + x(1)*(c(16) + x(2)*(c(31) + x(2)*c(52)) + x(1)*(c(30) + x(2)*c(51) + x(1)*c(50))))) + x(0)*(c(1) + x(2)*(c(6) + x(2)*(c(15) + x(2)*(c(29) + x(2)*c(49)))) + x(1)*(c(5) + x(2)*(c(14) + x(2)*(c(28) + x(2)*c(48))) + x(1)*(c(13) + x(2)*(c(27) + x(2)*c(47)) + x(1)*(c(26) + x(2)*c(46) + x(1)*c(45)))) + x(0)*(c(4) + x(2)*(c(12) + x(2)*(c(25) + x(2)*c(44))) + x(1)*(c(11) + x(2)*(c(24) + x(2)*c(43)) + x(1)*(c(23) + x(2)*c(42) + x(1)*c(41))) + x(0)*(c(10) + x(2)*(c(22) + x(2)*c(40)) + x(1)*(c(21) + x(2)*c(39) + x(1)*c(38)) + x(0)*(c(20) + x(2)*c(37) + x(1)*c(36) + x(0)*c(35)))));
            }

            TinyVector<double,3> grad(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,3>(c(1) + x(2)*(c(6) + x(2)*(c(15) + x(2)*(c(29) + x(2)*c(49)))) + x(1)*(c(5) + x(2)*(c(14) + x(2)*(c(28) + x(2)*c(48))) + x(1)*(c(13) + x(2)*(c(27) + x(2)*c(47)) + x(1)*(c(26) + x(2)*c(46) + x(1)*c(45)))) + x(0)*(2.0*c(4) + x(2)*(2.0*c(12) + x(2)*(2.0*c(25) + x(2)*(2.0*c(44)))) + x(1)*(2.0*c(11) + x(2)*(2.0*c(24) + x(2)*(2.0*c(43))) + x(1)*(2.0*c(23) + x(2)*(2.0*c(42)) + x(1)*(2.0*c(41)))) + x(0)*(3.0*c(10) + x(2)*(3.0*c(22) + x(2)*(3.0*c(40))) + x(1)*(3.0*c(21) + x(2)*(3.0*c(39)) + x(1)*(3.0*c(38))) + x(0)*(4.0*c(20) + x(2)*(4.0*c(37)) + x(1)*(4.0*c(36)) + x(0)*(5.0*c(35))))),
                    c(2) + x(2)*(c(8) + x(2)*(c(18) + x(2)*(c(33) + x(2)*c(54)))) + x(1)*(2.0*c(7) + x(2)*(2.0*c(17) + x(2)*(2.0*c(32) + x(2)*(2.0*c(53)))) + x(1)*(3.0*c(16) + x(2)*(3.0*c(31) + x(2)*(3.0*c(52))) + x(1)*(4.0*c(30) + x(2)*(4.0*c(51)) + x(1)*(5.0*c(50))))) + x(0)*(c(5) + x(2)*(c(14) + x(2)*(c(28) + x(2)*c(48))) + x(1)*(2.0*c(13) + x(2)*(2.0*c(27) + x(2)*(2.0*c(47))) + x(1)*(3.0*c(26) + x(2)*(3.0*c(46)) + x(1)*(4.0*c(45)))) + x(0)*(c(11) + x(2)*(c(24) + x(2)*c(43)) + x(1)*(2.0*c(23) + x(2)*(2.0*c(42)) + x(1)*(3.0*c(41))) + x(0)*(c(21) + x(2)*c(39) + x(1)*(2.0*c(38)) + x(0)*c(36)))),
                    c(3) + x(2)*(2.0*c(9) + x(2)*(3.0*c(19) + x(2)*(4.0*c(34) + x(2)*(5.0*c(55))))) + x(1)*(c(8) + x(2)*(2.0*c(18) + x(2)*(3.0*c(33) + x(2)*(4.0*c(54)))) + x(1)*(c(17) + x(2)*(2.0*c(32) + x(2)*(3.0*c(53))) + x(1)*(c(31) + x(2)*(2.0*c(52)) + x(1)*c(51)))) + x(0)*(c(6) + x(2)*(2.0*c(15) + x(2)*(3.0*c(29) + x(2)*(4.0*c(49)))) + x(1)*(c(14) + x(2)*(2.0*c(28) + x(2)*(3.0*c(48))) + x(1)*(c(27) + x(2)*(2.0*c(47)) + x(1)*c(46))) + x(0)*(c(12) + x(2)*(2.0*c(25) + x(2)*(3.0*c(44))) + x(1)*(c(24) + x(2)*(2.0*c(43)) + x(1)*c(42)) + x(0)*(c(22) + x(2)*(2.0*c(40)) + x(1)*c(39) + x(0)*c(37)))));
            }

            TinyVector<double,6> hessian(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,6>(2.0*c(4) + x(2)*(2.0*c(12) + x(2)*(2.0*c(25) + x(2)*(2.0*c(44)))) + x(1)*(2.0*c(11) + x(2)*(2.0*c(24) + x(2)*(2.0*c(43))) + x(1)*(2.0*c(23) + x(2)*(2.0*c(42)) + x(1)*(2.0*c(41)))) + x(0)*(6.0*c(10) + x(2)*(6.0*c(22) + x(2)*(6.0*c(40))) + x(1)*(6.0*c(21) + x(2)*(6.0*c(39)) + x(1)*(6.0*c(38))) + x(0)*(12.0*c(20) + x(2)*(12.0*c(37)) + x(1)*(12.0*c(36)) + x(0)*(20.0*c(35)))),
                    c(5) + x(2)*(c(14) + x(2)*(c(28) + x(2)*c(48))) + x(1)*(2.0*c(13) + x(2)*(2.0*c(27) + x(2)*(2.0*c(47))) + x(1)*(3.0*c(26) + x(2)*(3.0*c(46)) + x(1)*(4.0*c(45)))) + x(0)*(2.0*c(11) + x(2)*(2.0*c(24) + x(2)*(2.0*c(43))) + x(1)*(4.0*c(23) + x(2)*(4.0*c(42)) + x(1)*(6.0*c(41))) + x(0)*(3.0*c(21) + x(2)*(3.0*c(39)) + x(1)*(6.0*c(38)) + x(0)*(4.0*c(36)))),
                    c(6) + x(2)*(2.0*c(15) + x(2)*(3.0*c(29) + x(2)*(4.0*c(49)))) + x(1)*(c(14) + x(2)*(2.0*c(28) + x(2)*(3.0*c(48))) + x(1)*(c(27) + x(2)*(2.0*c(47)) + x(1)*c(46))) + x(0)*(2.0*c(12) + x(2)*(4.0*c(25) + x(2)*(6.0*c(44))) + x(1)*(2.0*c(24) + x(2)*(4.0*c(43)) + x(1)*(2.0*c(42))) + x(0)*(3.0*c(22) + x(2)*(6.0*c(40)) + x(1)*(3.0*c(39)) + x(0)*(4.0*c(37)))),
                    2.0*c(7) + x(2)*(2.0*c(17) + x(2)*(2.0*c(32) + x(2)*(2.0*c(53)))) + x(1)*(6.0*c(16) + x(2)*(6.0*c(31) + x(2)*(6.0*c(52))) + x(1)*(12.0*c(30) + x(2)*(12.0*c(51)) + x(1)*(20.0*c(50)))) + x(0)*(2.0*c(13) + x(2)*(2.0*c(27) + x(2)*(2.0*c(47))) + x(1)*(6.0*c(26) + x(2)*(6.0*c(46)) + x(1)*(12.0*c(45))) + x(0)*(2.0*c(23) + x(2)*(2.0*c(42)) + x(1)*(6.0*c(41)) + x(0)*(2.0*c(38)))),
                    c(8) + x(2)*(2.0*c(18) + x(2)*(3.0*c(33) + x(2)*(4.0*c(54)))) + x(1)*(2.0*c(17) + x(2)*(4.0*c(32) + x(2)*(6.0*c(53))) + x(1)*(3.0*c(31) + x(2)*(6.0*c(52)) + x(1)*(4.0*c(51)))) + x(0)*(c(14) + x(2)*(2.0*c(28) + x(2)*(3.0*c(48))) + x(1)*(2.0*c(27) + x(2)*(4.0*c(47)) + x(1)*(3.0*c(46))) + x(0)*(c(24) + x(2)*(2.0*c(43)) + x(1)*(2.0*c(42)) + x(0)*c(39))),
                    2.0*c(9) + x(2)*(6.0*c(19) + x(2)*(12.0*c(34) + x(2)*(20.0*c(55)))) + x(1)*(2.0*c(18) + x(2)*(6.0*c(33) + x(2)*(12.0*c(54))) + x(1)*(2.0*c(32) + x(2)*(6.0*c(53)) + x(1)*(2.0*c(52)))) + x(0)*(2.0*c(15) + x(2)*(6.0*c(29) + x(2)*(12.0*c(49))) + x(1)*(2.0*c(28) + x(2)*(6.0*c(48)) + x(1)*(2.0*c(47))) + x(0)*(2.0*c(25) + x(2)*(6.0*c(44)) + x(1)*(2.0*c(43)) + x(0)*(2.0*c(40)))));
            }
        };

        /* Three-dimensional, tricubic interpolant guaranteeing C^1 continuity between cells on a rectangular
            grid, using second order finite differences evaluated at gird points to ensure continuity of
            the gradient across grid cells. This interpolant is the analogue of the bicubic in two dimensions. */
        struct N3_Tricubic
        {
            enum { order = 3 };

            TinyVector<double,64> c;

            template<typename F>
            N3_Tricubic(const TinyVector<int,3>& i, const F& phi, const TinyVector<double,3>& dx)
            {
                TinyVector<int,3> stencilext = 4;
                blitz::Array<double,3> stencil(stencilext);

                for (MultiLoop<3> b(0, 4); b; ++b)
                {
                    TinyVector<int,3> j = i - 1 + b();
                    stencil(b()) = phi(j);
                }

                const double* rhs = stencil.data();
                for (int i = 0; i < 64; ++i)
                {
                    c(i) = 0.0;
                    for (int j = 0; j < 64; ++j)
                        c(i) += detail::StencilPolyData::N3_tricubicInverse()[i*64+j]*rhs[j];
                }

                TinyVector<double,3> dxinv = 1.0 / dx;
                double sx = 1.0;
                for (int i = 0; i < 4; ++i)
                {
                    double sy = 1.0;
                    for (int j = 0; j < 4; ++j)
                    {
                        double sz = 1.0;
                        for (int k = 0; k < 4; ++k)
                        {
                            c(i*16+j*4+k) *= sx*sy*sz;
                            sz *= dxinv(2);
                        }
                        sy *= dxinv(1);
                    }
                    sx *= dxinv(0);
                }
            }

            double operator() (const TinyVector<double,3>& x) const
            {
                return c(0) + x(2)*(c(1) + x(2)*(c(2) + x(2)*c(3))) + x(1)*(c(4) + x(2)*(c(5) + x(2)*(c(6) + x(2)*c(7))) + x(1)*(c(8) + x(2)*(c(9) + x(2)*(c(10) + x(2)*c(11))) + x(1)*(c(12) + x(2)*(c(13) + x(2)*(c(14) + x(2)*c(15)))))) + x(0)*(c(16) + x(2)*(c(17) + x(2)*(c(18) + x(2)*c(19))) + x(1)*(c(20) + x(2)*(c(21) + x(2)*(c(22) + x(2)*c(23))) + x(1)*(c(24) + x(2)*(c(25) + x(2)*(c(26) + x(2)*c(27))) + x(1)*(c(28) + x(2)*(c(29) + x(2)*(c(30) + x(2)*c(31)))))) + x(0)*(c(32) + x(2)*(c(33) + x(2)*(c(34) + x(2)*c(35))) + x(1)*(c(36) + x(2)*(c(37) + x(2)*(c(38) + x(2)*c(39))) + x(1)*(c(40) + x(2)*(c(41) + x(2)*(c(42) + x(2)*c(43))) + x(1)*(c(44) + x(2)*(c(45) + x(2)*(c(46) + x(2)*c(47)))))) + x(0)*(c(48) + x(2)*(c(49) + x(2)*(c(50) + x(2)*c(51))) + x(1)*(c(52) + x(2)*(c(53) + x(2)*(c(54) + x(2)*c(55))) + x(1)*(c(56) + x(2)*(c(57) + x(2)*(c(58) + x(2)*c(59))) + x(1)*(c(60) + x(2)*(c(61) + x(2)*(c(62) + x(2)*c(63)))))))));
            }

            TinyVector<double,3> grad(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,3>(c(16) + x(2)*(c(17) + x(2)*(c(18) + x(2)*c(19))) + x(1)*(c(20) + x(2)*(c(21) + x(2)*(c(22) + x(2)*c(23))) + x(1)*(c(24) + x(2)*(c(25) + x(2)*(c(26) + x(2)*c(27))) + x(1)*(c(28) + x(2)*(c(29) + x(2)*(c(30) + x(2)*c(31)))))) + x(0)*(2.0*c(32) + x(2)*(2.0*c(33) + x(2)*(2.0*c(34) + x(2)*(2.0*c(35)))) + x(1)*(2.0*c(36) + x(2)*(2.0*c(37) + x(2)*(2.0*c(38) + x(2)*(2.0*c(39)))) + x(1)*(2.0*c(40) + x(2)*(2.0*c(41) + x(2)*(2.0*c(42) + x(2)*(2.0*c(43)))) + x(1)*(2.0*c(44) + x(2)*(2.0*c(45) + x(2)*(2.0*c(46) + x(2)*(2.0*c(47))))))) + x(0)*(3.0*c(48) + x(2)*(3.0*c(49) + x(2)*(3.0*c(50) + x(2)*(3.0*c(51)))) + x(1)*(3.0*c(52) + x(2)*(3.0*c(53) + x(2)*(3.0*c(54) + x(2)*(3.0*c(55)))) + x(1)*(3.0*c(56) + x(2)*(3.0*c(57) + x(2)*(3.0*c(58) + x(2)*(3.0*c(59)))) + x(1)*(3.0*c(60) + x(2)*(3.0*c(61) + x(2)*(3.0*c(62) + x(2)*(3.0*c(63))))))))),
                    c(4) + x(2)*(c(5) + x(2)*(c(6) + x(2)*c(7))) + x(1)*(2.0*c(8) + x(2)*(2.0*c(9) + x(2)*(2.0*c(10) + x(2)*(2.0*c(11)))) + x(1)*(3.0*c(12) + x(2)*(3.0*c(13) + x(2)*(3.0*c(14) + x(2)*(3.0*c(15)))))) + x(0)*(c(20) + x(2)*(c(21) + x(2)*(c(22) + x(2)*c(23))) + x(1)*(2.0*c(24) + x(2)*(2.0*c(25) + x(2)*(2.0*c(26) + x(2)*(2.0*c(27)))) + x(1)*(3.0*c(28) + x(2)*(3.0*c(29) + x(2)*(3.0*c(30) + x(2)*(3.0*c(31)))))) + x(0)*(c(36) + x(2)*(c(37) + x(2)*(c(38) + x(2)*c(39))) + x(1)*(2.0*c(40) + x(2)*(2.0*c(41) + x(2)*(2.0*c(42) + x(2)*(2.0*c(43)))) + x(1)*(3.0*c(44) + x(2)*(3.0*c(45) + x(2)*(3.0*c(46) + x(2)*(3.0*c(47)))))) + x(0)*(c(52) + x(2)*(c(53) + x(2)*(c(54) + x(2)*c(55))) + x(1)*(2.0*c(56) + x(2)*(2.0*c(57) + x(2)*(2.0*c(58) + x(2)*(2.0*c(59)))) + x(1)*(3.0*c(60) + x(2)*(3.0*c(61) + x(2)*(3.0*c(62) + x(2)*(3.0*c(63))))))))),
                    c(1) + x(2)*(2.0*c(2) + x(2)*(3.0*c(3))) + x(1)*(c(5) + x(2)*(2.0*c(6) + x(2)*(3.0*c(7))) + x(1)*(c(9) + x(2)*(2.0*c(10) + x(2)*(3.0*c(11))) + x(1)*(c(13) + x(2)*(2.0*c(14) + x(2)*(3.0*c(15)))))) + x(0)*(c(17) + x(2)*(2.0*c(18) + x(2)*(3.0*c(19))) + x(1)*(c(21) + x(2)*(2.0*c(22) + x(2)*(3.0*c(23))) + x(1)*(c(25) + x(2)*(2.0*c(26) + x(2)*(3.0*c(27))) + x(1)*(c(29) + x(2)*(2.0*c(30) + x(2)*(3.0*c(31)))))) + x(0)*(c(33) + x(2)*(2.0*c(34) + x(2)*(3.0*c(35))) + x(1)*(c(37) + x(2)*(2.0*c(38) + x(2)*(3.0*c(39))) + x(1)*(c(41) + x(2)*(2.0*c(42) + x(2)*(3.0*c(43))) + x(1)*(c(45) + x(2)*(2.0*c(46) + x(2)*(3.0*c(47)))))) + x(0)*(c(49) + x(2)*(2.0*c(50) + x(2)*(3.0*c(51))) + x(1)*(c(53) + x(2)*(2.0*c(54) + x(2)*(3.0*c(55))) + x(1)*(c(57) + x(2)*(2.0*c(58) + x(2)*(3.0*c(59))) + x(1)*(c(61) + x(2)*(2.0*c(62) + x(2)*(3.0*c(63))))))))));
            }

            TinyVector<double,6> hessian(const TinyVector<double,3>& x) const
            {
                return TinyVector<double,6>(2.0*c(32) + x(2)*(2.0*c(33) + x(2)*(2.0*c(34) + x(2)*(2.0*c(35)))) + x(1)*(2.0*c(36) + x(2)*(2.0*c(37) + x(2)*(2.0*c(38) + x(2)*(2.0*c(39)))) + x(1)*(2.0*c(40) + x(2)*(2.0*c(41) + x(2)*(2.0*c(42) + x(2)*(2.0*c(43)))) + x(1)*(2.0*c(44) + x(2)*(2.0*c(45) + x(2)*(2.0*c(46) + x(2)*(2.0*c(47))))))) + x(0)*(6.0*c(48) + x(2)*(6.0*c(49) + x(2)*(6.0*c(50) + x(2)*(6.0*c(51)))) + x(1)*(6.0*c(52) + x(2)*(6.0*c(53) + x(2)*(6.0*c(54) + x(2)*(6.0*c(55)))) + x(1)*(6.0*c(56) + x(2)*(6.0*c(57) + x(2)*(6.0*c(58) + x(2)*(6.0*c(59)))) + x(1)*(6.0*c(60) + x(2)*(6.0*c(61) + x(2)*(6.0*c(62) + x(2)*(6.0*c(63)))))))),
                    c(20) + x(2)*(c(21) + x(2)*(c(22) + x(2)*c(23))) + x(1)*(2.0*c(24) + x(2)*(2.0*c(25) + x(2)*(2.0*c(26) + x(2)*(2.0*c(27)))) + x(1)*(3.0*c(28) + x(2)*(3.0*c(29) + x(2)*(3.0*c(30) + x(2)*(3.0*c(31)))))) + x(0)*(2.0*c(36) + x(2)*(2.0*c(37) + x(2)*(2.0*c(38) + x(2)*(2.0*c(39)))) + x(1)*(4.0*c(40) + x(2)*(4.0*c(41) + x(2)*(4.0*c(42) + x(2)*(4.0*c(43)))) + x(1)*(6.0*c(44) + x(2)*(6.0*c(45) + x(2)*(6.0*c(46) + x(2)*(6.0*c(47)))))) + x(0)*(3.0*c(52) + x(2)*(3.0*c(53) + x(2)*(3.0*c(54) + x(2)*(3.0*c(55)))) + x(1)*(6.0*c(56) + x(2)*(6.0*c(57) + x(2)*(6.0*c(58) + x(2)*(6.0*c(59)))) + x(1)*(9.0*c(60) + x(2)*(9.0*c(61) + x(2)*(9.0*c(62) + x(2)*(9.0*c(63)))))))),
                    c(17) + x(2)*(2.0*c(18) + x(2)*(3.0*c(19))) + x(1)*(c(21) + x(2)*(2.0*c(22) + x(2)*(3.0*c(23))) + x(1)*(c(25) + x(2)*(2.0*c(26) + x(2)*(3.0*c(27))) + x(1)*(c(29) + x(2)*(2.0*c(30) + x(2)*(3.0*c(31)))))) + x(0)*(2.0*c(33) + x(2)*(4.0*c(34) + x(2)*(6.0*c(35))) + x(1)*(2.0*c(37) + x(2)*(4.0*c(38) + x(2)*(6.0*c(39))) + x(1)*(2.0*c(41) + x(2)*(4.0*c(42) + x(2)*(6.0*c(43))) + x(1)*(2.0*c(45) + x(2)*(4.0*c(46) + x(2)*(6.0*c(47)))))) + x(0)*(3.0*c(49) + x(2)*(6.0*c(50) + x(2)*(9.0*c(51))) + x(1)*(3.0*c(53) + x(2)*(6.0*c(54) + x(2)*(9.0*c(55))) + x(1)*(3.0*c(57) + x(2)*(6.0*c(58) + x(2)*(9.0*c(59))) + x(1)*(3.0*c(61) + x(2)*(6.0*c(62) + x(2)*(9.0*c(63)))))))),
                    2.0*c(8) + x(2)*(2.0*c(9) + x(2)*(2.0*c(10) + x(2)*(2.0*c(11)))) + x(1)*(6.0*c(12) + x(2)*(6.0*c(13) + x(2)*(6.0*c(14) + x(2)*(6.0*c(15))))) + x(0)*(2.0*c(24) + x(2)*(2.0*c(25) + x(2)*(2.0*c(26) + x(2)*(2.0*c(27)))) + x(1)*(6.0*c(28) + x(2)*(6.0*c(29) + x(2)*(6.0*c(30) + x(2)*(6.0*c(31))))) + x(0)*(2.0*c(40) + x(2)*(2.0*c(41) + x(2)*(2.0*c(42) + x(2)*(2.0*c(43)))) + x(1)*(6.0*c(44) + x(2)*(6.0*c(45) + x(2)*(6.0*c(46) + x(2)*(6.0*c(47))))) + x(0)*(2.0*c(56) + x(2)*(2.0*c(57) + x(2)*(2.0*c(58) + x(2)*(2.0*c(59)))) + x(1)*(6.0*c(60) + x(2)*(6.0*c(61) + x(2)*(6.0*c(62) + x(2)*(6.0*c(63)))))))),
                    c(5) + x(2)*(2.0*c(6) + x(2)*(3.0*c(7))) + x(1)*(2.0*c(9) + x(2)*(4.0*c(10) + x(2)*(6.0*c(11))) + x(1)*(3.0*c(13) + x(2)*(6.0*c(14) + x(2)*(9.0*c(15))))) + x(0)*(c(21) + x(2)*(2.0*c(22) + x(2)*(3.0*c(23))) + x(1)*(2.0*c(25) + x(2)*(4.0*c(26) + x(2)*(6.0*c(27))) + x(1)*(3.0*c(29) + x(2)*(6.0*c(30) + x(2)*(9.0*c(31))))) + x(0)*(c(37) + x(2)*(2.0*c(38) + x(2)*(3.0*c(39))) + x(1)*(2.0*c(41) + x(2)*(4.0*c(42) + x(2)*(6.0*c(43))) + x(1)*(3.0*c(45) + x(2)*(6.0*c(46) + x(2)*(9.0*c(47))))) + x(0)*(c(53) + x(2)*(2.0*c(54) + x(2)*(3.0*c(55))) + x(1)*(2.0*c(57) + x(2)*(4.0*c(58) + x(2)*(6.0*c(59))) + x(1)*(3.0*c(61) + x(2)*(6.0*c(62) + x(2)*(9.0*c(63)))))))),
                    2.0*c(2) + x(2)*(6.0*c(3)) + x(1)*(2.0*c(6) + x(2)*(6.0*c(7)) + x(1)*(2.0*c(10) + x(2)*(6.0*c(11)) + x(1)*(2.0*c(14) + x(2)*(6.0*c(15))))) + x(0)*(2.0*c(18) + x(2)*(6.0*c(19)) + x(1)*(2.0*c(22) + x(2)*(6.0*c(23)) + x(1)*(2.0*c(26) + x(2)*(6.0*c(27)) + x(1)*(2.0*c(30) + x(2)*(6.0*c(31))))) + x(0)*(2.0*c(34) + x(2)*(6.0*c(35)) + x(1)*(2.0*c(38) + x(2)*(6.0*c(39)) + x(1)*(2.0*c(42) + x(2)*(6.0*c(43)) + x(1)*(2.0*c(46) + x(2)*(6.0*c(47))))) + x(0)*(2.0*c(50) + x(2)*(6.0*c(51)) + x(1)*(2.0*c(54) + x(2)*(6.0*c(55)) + x(1)*(2.0*c(58) + x(2)*(6.0*c(59)) + x(1)*(2.0*c(62) + x(2)*(6.0*c(63)))))))));
            }
        };
    } // namespace detail
    // The following templated typedefs allow a particular dimension N and particular polynomial
    // specified by an integer parameter Degree to be translated to the specific polynomial
    // class implemented above. Degree = -1 means use the bicubic/tricubic. If one tries to
    // use an underfined polynomial, a compiler error relating to "InvalidPoly" will be generated.

    struct InvalidPoly {};

    template<int N, int Degree>
    struct StencilPoly
    {
        typedef InvalidPoly T_Poly;
    };

    template<>
    struct StencilPoly<2,2>
    {
        typedef detail::N2_PolyDegree2 T_Poly;
    };

    template<>
    struct StencilPoly<2,3>
    {
        typedef detail::N2_PolyDegree3 T_Poly;
    };

    template<>
    struct StencilPoly<2,4>
    {
        typedef detail::N2_PolyDegree4 T_Poly;
    };

    template<>
    struct StencilPoly<2,5>
    {
        typedef detail::N2_PolyDegree5 T_Poly;
    };

    template<>
    struct StencilPoly<2,-1>
    {
        typedef detail::N2_Bicubic T_Poly;
    };

    template<>
    struct StencilPoly<3,2>
    {
        typedef detail::N3_PolyDegree2 T_Poly;
    };

    template<>
    struct StencilPoly<3,3>
    {
        typedef detail::N3_PolyDegree3 T_Poly;
    };

    template<>
    struct StencilPoly<3,4>
    {
        typedef detail::N3_PolyDegree4 T_Poly;
    };

    template<>
    struct StencilPoly<3,5>
    {
        typedef detail::N3_PolyDegree5 T_Poly;
    };

    template<>
    struct StencilPoly<3,-1>
    {
        typedef detail::N3_Tricubic T_Poly;
    };
} // namespace Algoim

#endif
