// This code can be used to verify the convergence results described in the paper
//    R. I. Saye, High-order methods for computing distances to implicitly defined surfaces,
//    Communications in Applied Mathematics and Computational Science, 9(1), 107-141 (2014),
//    http://dx.doi.org/10.2140/camcos.2014.9.107
// It consists of several test interfaces (circle, sphere, square, cube, ellipse,
// ellipsoid, rounded rectangle, rounded cylinder) and a driver "executeTest" for
// running the test case on a range of grid sizes.

#include <cstring>
#include <algorithm>
#include <cassert>
#include "hocp.hpp"
#include "utility.hpp"

using algoim::uvector;
using algoim::util::sqr;
#define PI 3.1415926535897932384626433832795


/* Each of the following Test structs defines
    - phi(x): the level set function that implicitly-defines the interface
    - cp(x): the exact closest point function, that returns true iff near a shock based on a given tolerance */

// Circle or sphere with radius 0.5 centred at the origin
template<int N>
struct SphereTest
{
    double phi(const uvector<double,N>& x) const
    {
        // Although phi is the exact signed distance function to the sphere, it can only be 
        // approximated by the polynomial interpolants
        return norm(x) - 0.5;
    }

    bool cp(const uvector<double,N>& x, double tol, uvector<double,N>& cp) const
    {
        if (sqrnorm(x) == 0.0)
        {
            cp = 0.0;
            cp(0) = 0.5;
        }
        else
            cp = (0.5 / norm(x))*x;
        // There is only one shock, at the origin
        return norm(x) < tol;
    }
};

// Square or cube of width 1.0 centred at the origin
template<int N>
struct CubeTest
{
    double phi(const uvector<double,N>& x) const
    {
        // This function is the signed-distance function to the cube *only inside* the cube; outside the cube it 
        // is not the distance to the zero level set
        double res = std::numeric_limits<double>::max();
        for (int i = 0; i < N; ++i)
            res = std::min(res, 0.5 - std::abs(x(i)));
        return res;
    }

    bool cp(const uvector<double,N>& x, double tol, uvector<double,N>& cp) const
    {
        int count = 0;
        for (int i = 0; i < N; ++i)
            if (x(i) >= 0.5 || x(i) <= -0.5)
                ++count;
        // If outside the cube, the components of the closest point to x can be computed independently
        if (count > 0)
        {
            for (int i = 0; i < N; ++i)
                if (x(i) >= 0.5)
                    cp(i) = 0.5;
                else if (x(i) <= -0.5)
                    cp(i) = -0.5;
                else
                    cp(i) = x(i);
            return false;
        }
        // If inside the cube, find the closest face to x
        double dist = std::numeric_limits<double>::max(), d;
        for (int i = 0; i < N; ++i)
        {
            uvector<double,N> y = x;
            y(i) = -0.5;
            if ((d = norm(x - y)) < dist)
            {
                dist = d;
                cp = y;
            }
            y(i) = 0.5;
            if ((d = norm(x - y)) < dist)
            {
                dist = d;
                cp = y;
            }
        }
        // We are near a shock if inside the cube and near a diagonal
        count = 0;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                if (std::abs(std::abs(x(i)) - std::abs(x(j))) < tol)
                    ++count;
        return count != 0;
    }
};

// Ellipse with semi-major axis 1/2 and semi-minor axis 1/3 centred at the origin
struct EllipsoidTest2D
{
    std::vector<uvector<double,2>> points;
    static const int nsample = 256;

    double a, b;

    EllipsoidTest2D()
    {
        a = 0.5;
        b = 1.0/3.0;
        // In order to use newtonCP to compute the exact cp(x) function, generate some seed points
        for (int i = 0; i < nsample; ++i)
        {
            double theta = (i + 0.5) / nsample * 2.0 * PI;
            points.push_back(uvector<double,2>(a*cos(theta), b*sin(theta)));
        }
    }

    double phi(const uvector<double,2>& x) const
    {
        return (1.0 - exp(-1.0*(sqr(x(0) - 0.3) + sqr(x(1) - 0.3))))*(sqrt(sqr(x(0)/a) + sqr(x(1)/b)) - 1.0);
    }

    // The functions operator(), grad() and hessian() are used in newtonCP to compute machine-accurate cp(x)
    double operator() (const uvector<double,2>& x) const
    {
        return sqr(x(0)/a) + sqr(x(1)/b) - 1.0;
    }

    uvector<double,2> grad(const uvector<double,2>& x) const
    {
        return uvector<double,2>(2.0/a*x(0)/a, 2.0/b*x(1)/b);
    }

    uvector<double,3> hessian(const uvector<double,2>& x) const
    {
        return uvector<double,3>(2.0/a/a, 0.0, 2.0/b/b);
    }

    bool cp(const uvector<double,2>& x, double tol, uvector<double,2>& cp) const
    {
        // Simple linear search on points to find the closest one; could speed this up with kdtree or something else
        // but performance is not of concern for these convergence tests
        double mind = std::numeric_limits<double>::max();
        int ind = -1;
        for (int i = 0; i < nsample; ++i)
        {
            double d = sqrnorm(points[i] - x);
            if (d < mind)
            {
                mind = d;
                ind = i;
            }
        }
        assert(ind >= 0);
        assert(a > b);

        // Use this as an initial guess to a machine-accurate Newton procedure
        cp = points[ind];
        int result = algoim::newtonCP<2,EllipsoidTest2D>(cp, x, *this, 4.0*PI*a/nsample, 1e-30, 30);
        if (result < 0)
        {
            // We're likely at a perfect curvature singularity (to machine precision). This does not happen in the tests considered here.
            std::cout << "newtonCP failed on EllipsoidTest2D, x = " << x << std::endl;
            exit(0);
            cp = points[ind];
            return true;
        }

        // Check if we are near a shock
        double rad = b*b/a;
        if (std::abs(x(1)) < tol && std::abs(x(0)) <= a - rad + tol)
            return true;
        return false;
    }
};

// Ellipsoid with semi-principal axes 1/2, 1/3 and 1/2 centred at the origin
struct EllipsoidTest3D
{
    std::vector<uvector<double,3>> points;
    static const int nsample = 256;

    double a, b, c;

    EllipsoidTest3D()
    {
        a = 0.5;
        b = 1.0/3.0;
        c = 0.5;
        // In order to use newtonCP to compute exact cp(x) function, generate some seed points
        for (int i = 0; i < nsample; ++i) for (int j = 0; j < nsample; ++j)
        {
            double theta = (i + 0.5) / nsample * 2.0 * PI;
            double psi = (j + 0.5) / nsample * 2.0 * PI;
            points.push_back(uvector<double,3>(a*cos(theta), b*sin(theta)*cos(psi), c*sin(theta)*sin(psi)));
        }
    }

    double phi(const uvector<double,3>& x) const
    {
        return (1.0 - exp(-1.0*(sqr(x(0) - 0.3) + sqr(x(1) - 0.3))))*(sqrt(sqr(x(0)/a) + sqr(x(1)/b) + sqr(x(2)/c)) - 1.0);
    }

    // The functions operator(), grad() and hessian() are used by newtonCP to compute machine-accurate cp(x)
    double operator() (const uvector<double,3>& x) const
    {
        return sqr(x(0)/a) + sqr(x(1)/b) + sqr(x(2)/c) - 1.0;
    }

    uvector<double,3> grad(const uvector<double,3>& x) const
    {
        return uvector<double,3>(2.0/a*x(0)/a, 2.0/b*x(1)/b, 2.0/c*x(2)/c);
    }

    uvector<double,6> hessian(const uvector<double,3>& x) const
    {
        return uvector<double,6>(2.0/a/a, 0.0, 0.0, 2.0/b/b, 0.0, 2.0/c/c);
    }

    bool cp(const uvector<double,3>& x, double tol, uvector<double,3>& cp) const
    {
        // Simple linear search on points to find the closest one; this could be made faster with e.g. kdtree,
        // but performance is not of concern for these convergence tests
        double mind = std::numeric_limits<double>::max();
        int ind = -1;
        for (int i = 0; i < nsample*nsample; ++i)
        {
            double d = sqrnorm(points[i] - x);
            if (d < mind)
            {
                mind = d;
                ind = i;
            }
        }
        assert(ind >= 0);
        assert(a >= c && c > b);

        // Use this as an initial guess to a machine-accurate Newton procedure
        cp = points[ind];
        int result = algoim::newtonCP<3,EllipsoidTest3D>(cp, x, *this, 8.0*PI*a/nsample, 1e-30, 30);
        if (result < 0)
        {
            // We're likely at a perfect curvature singularity (to machine precision). This does not happen in the tests considered here.
            std::cout << "newtonCP failed on EllipsoidTest3D, x = " << x << std::endl;
            exit(0);
            cp = points[ind];
            return true;
        }

        // Check if we are near a shock
        double rad = b*b/a;
        if (std::abs(x(1)) < tol && sqrt(sqr(x(0)) + sqr(x(2))) <= a - rad + tol)
            return true;
        return false;
    }
};

// A rectangle with circular ends (in 2D) or cylinder with spherical ends (in 3D)
template<int N>
struct RoundedPipeTest
{
    double phi(const uvector<double,N>& x) const
    {
        uvector<double,N> x0(0.0), x1(0.0), y(0.0);
        x0(0) = -0.25; x1(0) = 0.25; y(0) = x(0);

        uvector<double,N> c;
        cp(x, 0.0, c);

        bool inside = norm(x - x0) <= 0.25 || norm(x - x1) <= 0.25 || (x(0) >= -0.25 && x(0) <= 0.25 && norm(x - y) <= 0.25);
        return inside? -norm(x - c) : norm(x - c);
    }

    bool cp(const uvector<double,N>& x, double tol, uvector<double,N>& cp) const
    {
        if (x(0) >= -0.25 && x(0) <= 0.25)
        {
            double norm = 0.0;
            for (int i = 1; i < N; ++i)
                norm += sqr(x(i));
            norm = sqrt(norm);
            if (norm == 0.0)
            {
                cp = 0.0;
                cp(1) = 0.25;
                cp(0) = x(0);
            }
            else
            {
                cp = (0.25/norm)*x;
                cp(0) = x(0);
            }
            return norm < tol;
        }
        else if (x(0) <= -0.25)
        {
            uvector<double,N> y = x; y(0) += 0.25;
            cp = (0.25/norm(y))*y;
            cp(0) -= 0.25;
            return norm(y) < tol;
        }
        else
        {
            uvector<double,N> y = x; y(0) -= 0.25;
            cp = (0.25/norm(y))*y;
            cp(0) += 0.25;
            return norm(y) < tol;
        }
    }
};

// A simple test functor whose purpose is to simulate a grid-defined scalar array
template<int N, typename Test>
struct TestFunctor
{
    const Test& test;
    const uvector<double,N> dx;
    const uvector<double,N> xmin; // xmin = coordinates of grid-point i = 0
    TestFunctor(const Test& test, const uvector<double,N>& dx, const uvector<double,N>& xmin) : test(test), dx(dx), xmin(xmin) {}
    double operator() (const uvector<int,N>& i) const
    {
        return test.phi(i*dx + xmin);
    }
};

/* Test engine */

struct TestResult
{
    double local_dist_l1, local_dist_lmax, local_cp_l1, local_cp_lmax;
    double global_dist_l1, global_dist_lmax, global_cp_l1, global_cp_lmax;
};

template<int N, int Degree, typename Test>
TestResult executeTest(int n, double domainLen, const Test& test)
{
    // Determine the type of polynomial to use based on given Degree and dimension N
    typedef typename algoim::StencilPoly<N,Degree>::T_Poly Poly;

    // The domain is [-domainLen/2, domainLen/2] and is discretised with a cell-centered grid with
    // n grid points in each dimension
    uvector<int,N> ext = n;
    uvector<double,N> dx = domainLen/n;
    uvector<double,N> xmin = -0.5*domainLen + 0.5*dx;

    // Create a functor whose purpose is to simulate a n-dimensional scalar array
    TestFunctor<N,Test> functor(test, dx, xmin);

    // Find all cells containing the interface and construct the high-order polynomials
    std::vector<algoim::detail::CellPoly<N,Poly>> cells;
    algoim::detail::createCellPolynomials(ext, functor, dx, false, cells);

    // Using the polynomials, sample the zero level set in each cell to create a cloud of seed points
    std::vector<uvector<double,N>> points;
    std::vector<int> pointcells;
    algoim::detail::samplePolynomials(cells, 2, dx, xmin, points, pointcells);

    // Construct a k-d tree from the seed points
    algoim::KDTree<double,N> kdtree(points);

    // Pass everything to the closest point computation engine
    algoim::ComputeHighOrderCP<N,Poly> hocp(std::numeric_limits<double>::max(), // bandradius = infinity
        0.5*max(dx), // amount of overlap, i.e. size of bounding ball in Newton's method
        sqr(std::max(1.0e-14, pow(max(dx), Poly::order))), // tolerance to determine convergence
        cells, kdtree, points, pointcells, dx, xmin);

    // Tolerance for deciding when near a shock
    double shockTol = 0.51*max(dx);

    // Radius of narrow band for 'local' error measurement
    double r_narrowband = 8.0*min(dx);

    TestResult result;
    result.local_dist_l1 = result.local_dist_lmax = result.local_cp_l1 = result.local_cp_lmax = 0.0;
    result.global_dist_l1 = result.global_dist_lmax = result.global_cp_l1 = result.global_cp_lmax = 0.0;

    uvector<int,4> counts = 0;

    // Loop over every grid point of domain
    for (algoim::MultiLoop<N> i(0, ext); ~i; ++i)
    {
        uvector<double,N> x = i()*dx + xmin;
        uvector<double,N> cp;

        // Compute the closest point to x
        hocp.compute(x, cp);

        // Calculate the exact closest point to x
        uvector<double,N> cpexact;
        bool atShock = test.cp(x, shockTol, cpexact);

        // Error in implied distance function
        double errdist = norm(x - cp) - norm(x - cpexact); 

        // Error in closest point function
        double errcp = norm(cpexact - cp); 

        // If within narrow band, update local error measurements
        if (norm(x - cp) < r_narrowband)
        {
            result.local_dist_l1 += std::abs(errdist);
            result.local_dist_lmax = std::max(result.local_dist_lmax, std::abs(errdist));
            ++counts(0);
            if (!atShock)
            {
                result.local_cp_l1 += errcp;
                result.local_cp_lmax = std::max(result.local_cp_lmax, errcp);
                ++counts(1);
            }
        }

        // Update global error measurements
        result.global_dist_l1 += std::abs(errdist);
        result.global_dist_lmax = std::max(result.global_dist_lmax, std::abs(errdist));
        ++counts(2);
        if (!atShock)
        {
            result.global_cp_l1 += errcp;
            result.global_cp_lmax = std::max(result.global_cp_lmax, errcp);
            ++counts(3);
        }
    }

    result.local_dist_l1 /= counts(0);
    result.local_cp_l1 /= counts(1);
    result.global_dist_l1 /= counts(2);
    result.global_cp_l1 /= counts(3);

    return result;
}

// Run a sequence of convergence tests over a given range of grid sizes
template<int N, int Degree, typename Test>
void runConvergenceTest(int n0, int n1, const Test& test)
{
    std::vector<uvector<double,8>> results;
    for (int n = n0, i = 0; n <= n1; n *= 2, ++i)
    {
        TestResult result = executeTest<N,Degree,Test>(n, 1.5, test);
        results.push_back(uvector<double,8>(result.local_dist_l1, result.local_dist_lmax, result.local_cp_l1, result.local_cp_lmax, 
            result.global_dist_l1, result.global_dist_lmax, result.global_cp_l1, result.global_cp_lmax));

        printf("n = %4d : ", n);
        for (int j = 0; j < 8; ++j)
        {
            printf("%.5e ", results[i](j));
            if (i > 0)
            {
                double rate = std::log(results[i-1](j) / results[i](j)) / std::log(2.0);
                if (rate >= 0.0)
                    printf("(%.1f) ", rate);
                else
                    printf("(---) ");
            }
            else
                printf("      ");
        }
        std::cout << std::endl;
    }
}

// Run a sequence of convergence tests for a given test problem
template<int N, typename Test>
void runTest()
{
    Test test;

    int n0 = 64;
    int n1 = 512;

    std::cout << "Error in a narrow band                                                                 Global error" << std::endl;
    std::cout << "Distance L1, Linfty                              Closest point L1, Linfty              Distance L1, Linfty                   Closest point L1, Linfty" << std::endl;

    std::cout << "\nn-cubic" << std::endl;
    runConvergenceTest<N,-1,Test>(n0, n1, test);

    std::cout << "\nTaylor degree 2" << std::endl;
    runConvergenceTest<N,2,Test>(n0, n1, test);

    std::cout << "\nTaylor degree 3" << std::endl;
    runConvergenceTest<N,3,Test>(n0, n1, test);

    std::cout << "\nTaylor degree 4" << std::endl;
    runConvergenceTest<N,4,Test>(n0, n1, test);

    std::cout << "\nTaylor degree 5" << std::endl;
    runConvergenceTest<N,5,Test>(n0, n1, test);
}

#if ALGOIM_EXAMPLES_DRIVER == 0 || ALGOIM_EXAMPLES_DRIVER == 3
// Usage: exename N test, where N = 2 or 3, and test = sphere, ellipsoid, cube, pipe
int main(int argc, char* argv[])
{
    if (argc >= 3 && (strcmp(argv[1], "2") == 0 || strcmp(argv[1], "3") == 0))
    {
        // Convergence tests for all test problems
        int dim = 2;
        sscanf(argv[1], "%d", &dim);
        if (strcmp(argv[2], "sphere") == 0)
            dim == 2 ? runTest<2, SphereTest<2> >() : runTest<3, SphereTest<3> >();
        if (strcmp(argv[2], "ellipsoid") == 0)
            dim == 2 ? runTest<2, EllipsoidTest2D>() : runTest<3, EllipsoidTest3D>();
        if (strcmp(argv[2], "cube") == 0)
            dim == 2 ? runTest<2, CubeTest<2> >() : runTest<3, CubeTest<3> >();
        if (strcmp(argv[2], "pipe") == 0)
            dim == 2 ? runTest<2, RoundedPipeTest<2> >() : runTest<3, RoundedPipeTest<3> >();
    }
    return 0;
}
#endif
