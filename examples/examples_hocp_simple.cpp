// Provides a very simple demonstration of how to reinitialise a level set function with
// Algoim's high-order closest point algorithms. The file contains a single main() routine;
// compile it as you would for any .cpp file with a main() entry point.

#include <vector>
#include <algorithm>
#include "hocp.hpp"

// Two-dimensional example using a circle of radius 1
void test2d(int n)
{
    std::vector<double> phi_data(n*n), sdfexact_data(n*n);
    auto phi = [&](const algoim::uvector<int,2>& i) -> double& { return phi_data[i(0)*n + i(1)]; };
    auto sdfexact = [&](const algoim::uvector<int,2>& i) -> double& { return sdfexact_data[i(0)*n + i(1)]; };

    double dx = 4.0 / n;
    for (int i = 0; i < n; ++i)
    {
        double x = -2.0 + (i + 0.5) * dx;
        for (int j = 0; j < n; ++j)
        {
            double y = -2.0 + (j + 0.5) * dx;
            phi(algoim::uvector<int,2>{i,j}) = exp(x*x + y*y - 1.0) - 1.0;
            sdfexact(algoim::uvector<int,2>{i,j}) = sqrt(x*x + y*y) - 1.0;
        }
    }

    // Different polynomial interpolants can be used
    //algoim::reinit<2,-1>(phi, n, dx, 10.0);
    //algoim::reinit<2,2>(phi, n, dx, 10.0);
    algoim::reinit<2,3>(phi, n, dx, 10.0);
    //algoim::reinit<2,4>(phi, n, dx, 10.0);
    //algoim::reinit<2,5>(phi, n, dx, 10.0);

    double maxerr = 0.0;
    for (int i = 0; i < n*n; ++i)
        maxerr = std::max(maxerr, std::abs(phi_data[i] - sdfexact_data[i]));

    std::cout << "Infinity norm of error = " << maxerr << std::endl;
}

// Three-dimensional example using a sphere of radius 1
void test3d(int n)
{
    std::vector<double> phi_data(n*n*n), sdfexact_data(n*n*n);
    auto phi = [&](const algoim::uvector<int,3>& i) -> double& { return phi_data[i(0)*n*n + i(1)*n + i(2)]; };
    auto sdfexact = [&](const algoim::uvector<int,3>& i) -> double& { return sdfexact_data[i(0)*n*n + i(1)*n + i(2)]; };

    double dx = 4.0 / n;
    for (int i = 0; i < n; ++i)
    {
        double x = -2.0 + (i + 0.5) * dx;
        for (int j = 0; j < n; ++j)
        {
            double y = -2.0 + (j + 0.5) * dx;
            for (int k = 0; k < n; ++k)
            {
                double z = -2.0 + (k + 0.5) * dx;
                phi(algoim::uvector<int,3>{i,j,k}) = exp(x*x + y*y + z*z - 1.0) - 1.0;
                sdfexact(algoim::uvector<int,3>{i,j,k}) = sqrt(x*x + y*y + z*z) - 1.0;
            }
        }
    }

    // Different polynomial interpolants can be used
    //algoim::reinit<3,-1>(phi, n, dx, 10.0);
    //algoim::reinit<3,2>(phi, n, dx, 10.0);
    algoim::reinit<3,3>(phi, n, dx, 10.0);
    //algoim::reinit<3,4>(phi, n, dx, 10.0);
    //algoim::reinit<3,5>(phi, n, dx, 10.0);

    double maxerr = 0.0;
    for (int i = 0; i < n*n*n; ++i)
        maxerr = std::max(maxerr, std::abs(phi_data[i] - sdfexact_data[i]));

    std::cout << "Infinity norm of error = " << maxerr << std::endl;
}

#if ALGOIM_EXAMPLES_DRIVER == 0 || ALGOIM_EXAMPLES_DRIVER == 2
// A very simple driver
int main(int argc, char* argv[])
{
    while (true)
    {
        int dim = 0, n = 0;
        std::cout << "Enter the dimension N = 2 or N = 3: ";
        std::cin >> dim;
        if (dim != 2 && dim != 3) break;
        std::cout << "Enter number of grid points in each dimension: ";
        std::cin >> n;
        if (n < 2) break;
        if (dim == 2)
            test2d(n);
        else
            test3d(n);
    }
    return 0;
}
#endif
