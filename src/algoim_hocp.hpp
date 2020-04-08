#ifndef ALGOIM_HOCP_HPP
#define ALGOIM_HOCP_HPP

/* Contains the main drivers for the high-order closest point algorithm, mainly for rectangular
   Cartesian grids, as described in the paper
       R. I. Saye, High-order methods for computing distances to implicitly defined surfaces,
       Communications in Applied Mathematics and Computational Science, 9(1), 107-141 (2014),
       http://dx.doi.org/10.2140/camcos.2014.9.107 */

#include <vector>
#include "algoim_multiloop.hpp"
#include "algoim_kdtree.hpp"
#include "algoim_newtoncp.hpp"
#include "algoim_stencilpoly.hpp"

namespace Algoim
{
    namespace detail
    {
        // A CellPoly struct contains the interpolated polynomial for a particular grid cell
        template<int N, typename Poly>
        struct CellPoly
        {
            Poly poly;                  // The polynomial
            TinyVector<int,N> i;        // Coordinates of the cell
            CellPoly(const Poly& poly, const TinyVector<int,N>& i) : poly(poly), i(i) {}
        };

        // PhiConstExtrap is a functor which implements a very simplistic treatment of boundary conditions: 
        // phi is extrapolated by a constant outside the domain (similar to how Neumann boundary conditions
        // are specified with ghost layers). In other words, if phi(i) is requested and i is outside the
        // domain, then the nearest grid point to i is used to define phi(i).
        template<int N>
        struct PhiConstExtrap
        {
            const blitz::Array<double,N>& phi;
            PhiConstExtrap(const blitz::Array<double,N>& phi) : phi(phi) {}
            double operator() (TinyVector<int,N> i) const
            {
                for (int dim = 0; dim < N; ++dim)
                {
                    if (i(dim) < 0) 
                        i(dim) = 0;
                    else if (i(dim) >= phi.extent(dim))
                        i(dim) = phi.extent(dim) - 1;
                }
                return phi(i);
            }
        };

        // Given a Cartesian grid-defined level set function phi, interpolate phi on each relevant grid cell to
        // create a collection of polynomials. If interpAllCells is true, then every grid cell in the domain
        // is interpolated, otherwise only those cells for which phi changes sign on the vertices are considered.
        template<int N, typename Phi, typename Poly>
        void createCellPolynomials(const TinyVector<int,N>& ext, const Phi& phi, const TinyVector<double,N>& dx, bool interpAllCells, std::vector<CellPoly<N,Poly>>& cells)
        {
            if (interpAllCells)
            {
                for (MultiLoop<N> i(0, ext - 1); i; ++i)
                    cells.push_back(CellPoly<N,Poly>(Poly(i(), phi, dx), i()));
            }
            else
            {
                for (MultiLoop<N> i(0, ext - 1); i; ++i)
                {
                    int s = 0;
                    for (MultiLoop<N> j(i(), i() + 2); j; ++j)
                        s += (phi(j()) >= 0.0) ? 1 : -1;
                    if (std::abs(s) < 1 << N)
                        cells.push_back(CellPoly<N,Poly>(Poly(i(), phi, dx), i()));
                }
            }
        }

        // Given a collection of cell-generated polynomials, sample their zero level set to generate the
        // seed points in the cloud, where subcellExt is the number of subcells to use per cell. The output
        // is a collection of points and the corresponding index of the polynomial from which they were generated.
        template<int N, typename Poly>
        void samplePolynomials(const std::vector<CellPoly<N,Poly>>& cells, int subcellExt, const TinyVector<double,N>& dx,
            const TinyVector<double,N>& xmin, std::vector<TinyVector<double,N>>& points, std::vector<int>& pointcells)
        {
            points.clear();
            pointcells.clear();
            // subcellsqrradius is the induced squared radius of each subcell (used to discard points if they go too far)
            double subcellsqrradius = sqr(1.25*mag(dx)*0.5/subcellExt);
            // tolerance 1% of subcell size
            double isotolsqr = std::max(1.0e-25, 1e-4*subcellsqrradius);
            for (size_t i = 0, len = cells.size(); i < len; ++i)
            {
                for (MultiLoop<N> j(0, subcellExt); j; ++j)
                {
                    TinyVector<double,N> x = (j() + 0.5)*dx/subcellExt;
                    TinyVector<double,N> y = x;
                    if (newtonIso(x, cells[i].poly, isotolsqr, 10) > 0 && magsqr<double,N>(x - y) < subcellsqrradius)
                    {
                        points.push_back(cells[i].i*dx + xmin + x);
                        pointcells.push_back(static_cast<int>(i));
                    }
                }
            }
        }
    }

    /* The ComputeHighOrderCP struct is the main engine for computing high-order accurate closest points.
       It accepts as parameters:
         - bandrsqr: squared radius of the narrow band
         - overlapr: radius of the bounding ball used in Newton's method
         - cptolsqr: tolerance used to determine convergence in Newton's method
         - cells: collection of polynomials
         - kdtree: tree containing the seed points
         - points: collection of seed points
         - pointcells: index of the polynomial corresponding to each seed point
         - dx: grid cell size
         - xmin: coordinate of lower-left vertex of cell 0 */
    template<int N, typename Poly>
    struct ComputeHighOrderCP
    {
        const double bandrsqr, overlapr, cptolsqr;
        const std::vector<detail::CellPoly<N,Poly>>& cells;
        const KDTree<double,N>& kdtree;
        const std::vector<TinyVector<double,N>>& points;
        const std::vector<int>& pointcells;
        const TinyVector<double,N> dx;
        const TinyVector<double,N> xmin;

        ComputeHighOrderCP(double bandrsqr, double overlapr, double cptolsqr, const std::vector<detail::CellPoly<N,Poly>>& cells,
            const KDTree<double,N>& kdtree, const std::vector<TinyVector<double,N>>& points, const std::vector<int>& pointcells,
            const TinyVector<double,N>& dx, const TinyVector<double,N>& xmin)
            : bandrsqr(bandrsqr), overlapr(overlapr), cptolsqr(cptolsqr), cells(cells), kdtree(kdtree), points(points), pointcells(pointcells), dx(dx), xmin(xmin)
        {}

        // Given x, compute the closest point to x; returns false if the closest seed point is outside
        // the narrow band
        bool compute(const TinyVector<double,N>& x, TinyVector<double,N>& cp, double* signedDist = 0) const
        {
            // Find the closest point in the cloud
            int index = kdtree.nearest(x, bandrsqr);

            // If there is no point within the defined band radius, return false
            if (index < 0)
                return false;

            // Fetch the polynomial corresponding to the closest point in the cloud and calculate the reference point
            // and initial guess of the closest point in the local coordinate system
            const detail::CellPoly<N,Poly>& cell = cells[pointcells[index]];
            TinyVector<double,N> ref = x - (cell.i*dx + xmin);
            cp = points[index] - (cell.i*dx + xmin);

            // Execute Newton's method
            newtonCP<N,Poly>(cp, ref, cell.poly, overlapr, cptolsqr, 20);

            // If the signed distance is also requested, the sign can (usually) be determined
            // according to alignment of the normal and polynomial gradient at the closest point
            if (signedDist)
            {
                TinyVector<double,N> grad = cell.poly.grad(cp);
                if (dot(ref - cp, grad) >= 0.0)
                    *signedDist = mag<double,N>(ref - x);
                else
                    *signedDist = -mag<double,N>(ref - x);
            }

            // Adjust the closest point such that it is in the global coordinate system
            cp += cell.i*dx + xmin;
            return true;
        }
    };

    // Reinitialiase a Cartesian grid-defined level set function phi in a narrow band of radius r using the
    // high-order closest point algorithm. The template parameter Degree specifies the type of
    // polynomial to use.
    template<int N, int Degree>
    void reinit(blitz::Array<double,N>& phi, const TinyVector<double,N>& dx, double r)
    {
        // Determine the type of polynomial to use based on given Degree and dimension N
        typedef typename Algoim::StencilPoly<N,Degree>::T_Poly Poly;

        // In this simple implementation of a reinitialisation algorithm, boundaries of phi are treated with a simple type of extrapolation
        detail::PhiConstExtrap<N> phiextrap(phi);

        // Find all cells containing the interface and construct the high-order polynomials
        std::vector<detail::CellPoly<N,Poly>> cells;
        detail::createCellPolynomials(phi.extent(), phiextrap, dx, false, cells);

        // Using the polynomials, sample the zero level set in each cell to create a cloud of seed points
        std::vector<TinyVector<double,N>> points;
        std::vector<int> pointcells;
        detail::samplePolynomials<N,Poly>(cells, 2, dx, 0.0, points, pointcells);

        // Construct a k-d tree from the seed points
        KDTree<double,N> kdtree(points);

        // Pass everything to the closest point computation engine
        ComputeHighOrderCP<N,Poly> hocp(r < std::numeric_limits<double>::max() ? r*r : std::numeric_limits<double>::max(), // squared bandradius
            0.5*max(dx), // amount that each polynomial overlaps / size of the bounding ball in Newton's method
            sqr(std::max(1.0e-14, std::pow(blitz::max(dx), Poly::order))), // tolerance to determine convergence
            cells, kdtree, points, pointcells, dx, 0.0);

        // For each grid point in the domain, compute the closest point and use this to re-define phi as the signed distance function
        for (MultiLoop<N> i(0, phi.extent()); i; ++i)
        {
            TinyVector<double,N> x = i()*dx, cp;
            if (hocp.compute(x, cp))
                phi(i()) = (phi(i()) >= 0.0)? mag<double,N>(x - cp) : -mag<double,N>(x - cp);
            else
                phi(i()) = (phi(i()) >= 0.0)? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
        }
    }
} // namespace Algoim

#endif
