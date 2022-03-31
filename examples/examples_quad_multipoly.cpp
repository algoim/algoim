// Examples to demonstrate Algoim's methods for computing high-order accurate quadrature schemes
// on multi-component domains implicitly-defined by (one or more) multivariate Bernstein
// polynomials. Additional examples are provided on the GitHub documentation page, 
// https://algoim.github.io/

#include <iostream>
#include <iomanip>
#include <fstream>
#include "quadrature_multipoly.hpp"

using namespace algoim;

// Driver method which takes a functor phi defining a single polynomial in the reference
// rectangle [xmin, xmax]^N, of Bernstein degree P, along with an integrand function,
// and performances a q-refinement convergence study, comparing the computed integral
// with the given exact answers, for 1 <= q <= qMax.
template<int N, typename Phi, typename F>
void qConv(const Phi& phi, real xmin, real xmax, uvector<int,N> P, const F& integrand, int qMax, real volume_exact, real surf_exact)
{
    // Construct Bernstein polynomial by mapping [0,1] onto bounding box [xmin,xmax]
    xarray<real,N> phipoly(nullptr, P);
    algoim_spark_alloc(real, phipoly);
    bernstein::bernsteinInterpolate<N>([&](const uvector<real,N>& x) { return phi(xmin + x * (xmax - xmin)); }, phipoly);

    // Build quadrature hierarchy
    ImplicitPolyQuadrature<N> ipquad(phipoly);

    // Functional to evaluate volume and surface integrals of given integrand
    real volume, surf;
    auto compute = [&](int q)
    {
        volume = 0.0;
        surf = 0.0;
        // compute volume integral over {phi < 0} using AutoMixed strategy
        ipquad.integrate(AutoMixed, q, [&](const uvector<real,N>& x, real w)
        {
            if (bernstein::evalBernsteinPoly(phipoly, x) < 0)
                volume += w * integrand(xmin + x * (xmax - xmin));
        });
        // compute surface integral over {phi == 0} using AutoMixed strategy
        ipquad.integrate_surf(AutoMixed, q, [&](const uvector<real,N>& x, real w, const uvector<real,N>& wn)
        {
            surf += w * integrand(xmin + x * (xmax - xmin));
        });
        // scale appropriately
        volume *= pow(xmax - xmin, N);
        surf *= pow(xmax - xmin, N - 1);
    };

    // Compute results for all q and output in a convergence table
    for (int q = 1; q <= qMax; ++q)
    {
        compute(q);
        std::cout << q << ' ' << volume << ' ' << surf << ' ' << std::abs(volume - volume_exact)/volume_exact << ' ' << std::abs(surf - surf_exact)/surf_exact << std::endl;
    }
}

// Given a set of quadrature points and weights, output them to an VTP XML file for visualisation
// purposes, e.g., using ParaView
template<int N>
void outputQuadratureRuleAsVtpXML(const std::vector<uvector<real,N+1>>& q, std::string fn)
{
    static_assert(N == 2 || N == 3, "outputQuadratureRuleAsVtpXML only supports 2D and 3D quadrature schemes");
    std::ofstream stream(fn);
    stream << "<?xml version=\"1.0\"?>\n";
    stream << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    stream << "<PolyData>\n";
    stream << "<Piece NumberOfPoints=\"" << q.size() << "\" NumberOfVerts=\"" << q.size() << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    stream << "<Points>\n";
    stream << "  <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";
    for (const auto& pt : q)
        stream << pt(0) << ' ' << pt(1) << ' ' << (N == 3 ? pt(2) : 0.0) << ' ';
    stream << "</DataArray>\n";
    stream << "</Points>\n";
    stream << "<Verts>\n";
    stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
    for (size_t i = 0; i < q.size(); ++i)
        stream << i << ' ';
    stream <<  "</DataArray>\n";
    stream << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
    for (size_t i = 1; i <= q.size(); ++i)
        stream << i << ' ';
    stream << "</DataArray>\n";
    stream << "</Verts>\n";
    stream << "<PointData Scalars=\"w\">\n";
    stream << "  <DataArray type=\"Float32\" Name=\"w\" NumberOfComponents=\"1\" format=\"ascii\">";
    for (const auto& pt : q)
        stream << pt(N) << ' ';
    stream << "</DataArray>\n";
    stream << "</PointData>\n";
    stream << "</Piece>\n";
    stream << "</PolyData>\n";
    stream << "</VTKFile>\n";
};

// Driver method which takes a functor phi defining a single polynomial in the reference
// rectangle [xmin, xmax]^N, of Bernstein degree P, builds a quadrature scheme with the
// given q, and outputs it for visualisation in a set of VTP XML files
template<int N, typename F>
void outputQuadScheme(const F& fphi, real xmin, real xmax, const uvector<int,N>& P, int q, std::string qfile)
{
    // Construct phi by mapping [0,1] onto bounding box [xmin,xmax]
    xarray<real,N> phi(nullptr, P);
    algoim_spark_alloc(real, phi);
    bernstein::bernsteinInterpolate<N>([&](const uvector<real,N>& x) { return fphi(xmin + x * (xmax - xmin)); }, phi);

    // Build quadrature hierarchy
    ImplicitPolyQuadrature<N> ipquad(phi);

    // Compute quadrature scheme and record the nodes & weights; phase0 corresponds to
    // {phi < 0}, phase1 corresponds to {phi > 0}, and surf corresponds to {phi == 0}.
    std::vector<uvector<real,N+1>> phase0, phase1, surf;
    ipquad.integrate(AutoMixed, q, [&](const uvector<real,N>& x, real w)
    {
        if (bernstein::evalBernsteinPoly(phi, x) < 0)
            phase0.push_back(add_component(x, N, w));
        else
            phase1.push_back(add_component(x, N, w));
    });
    ipquad.integrate_surf(AutoMixed, q, [&](const uvector<real,N>& x, real w, const uvector<real,N>& wn)
    {
        surf.push_back(add_component(x, N, w));
    });

    // output to file
    outputQuadratureRuleAsVtpXML<N>(phase0, qfile + "-phase0.vtp");
    outputQuadratureRuleAsVtpXML<N>(phase1, qfile + "-phase1.vtp");
    outputQuadratureRuleAsVtpXML<N>(surf, qfile + "-surf.vtp");
}

// Driver method which takes two phi functors defining two polynomials in the reference
// rectangle [xmin, xmax]^N, each of of Bernstein degree P, builds a quadrature scheme with the
// given q, and outputs it for visualisation in a set of VTP XML files
template<int N, typename F1, typename F2>
void outputQuadScheme(const F1& fphi1, const F2& fphi2, real xmin, real xmax, const uvector<int,N>& P, int q, std::string qfile)
{
    // Construct phi by mapping [0,1] onto bounding box [xmin,xmax]
    xarray<real,N> phi1(nullptr, P), phi2(nullptr, P);
    algoim_spark_alloc(real, phi1, phi2);
    bernstein::bernsteinInterpolate<N>([&](const uvector<real,N>& x) { return fphi1(xmin + x * (xmax - xmin)); }, phi1);
    bernstein::bernsteinInterpolate<N>([&](const uvector<real,N>& x) { return fphi2(xmin + x * (xmax - xmin)); }, phi2);

    // Build quadrature hierarchy
    ImplicitPolyQuadrature<N> ipquad(phi1, phi2);

    // Compute quadrature scheme and record the nodes & weights; one could examine the signs
    // of phi1 and phi2 in order to separate the nodes into different components, but for
    // simplicity they are agglomerated
    std::vector<uvector<real,N+1>> vol, surf;
    ipquad.integrate(AutoMixed, q, [&](const uvector<real,N>& x, real w)
    {
        vol.push_back(add_component(x, N, w));
    });
    ipquad.integrate_surf(AutoMixed, q, [&](const uvector<real,N>& x, real w, const uvector<real,N>& wn)
    {
        surf.push_back(add_component(x, N, w));
    });

    // output to a file
    outputQuadratureRuleAsVtpXML<N>(vol, qfile + "-vol.vtp");
    outputQuadratureRuleAsVtpXML<N>(surf, qfile + "-surf.vtp");
}

#if ALGOIM_EXAMPLES_DRIVER == 0 || ALGOIM_EXAMPLES_DRIVER == 4
int main(int argc, char* argv[])
{
    std::cout << "Algoim Examples - High-order quadrature algorithms for multi-component domains implicitly-defined\n";
    std::cout << "by (one or more) multivariate Bernstein polynomials\n\n";
    std::cout << std::scientific << std::setprecision(10);

    // q-convergence study for a 2D ellipse
    {
        auto ellipse = [](const uvector<real,2>& x)
        {
            return x(0)*x(0) + x(1)*x(1)*4 - 1;
        };
        auto integrand = [](const uvector<real,2>& x)
        {
            return 1.0;
        };
        real volume_exact = algoim::util::pi / 2;
        real surf_exact = 4.844224110273838099214251598195914705976959198943300412541558176231060;
        std::cout << "\n\nEllipse q-convergence test\n";
        std::cout << "q      area(q)         perim(q)        area error       perim error\n";
        qConv<2>(ellipse, -1.1, 1.1, 3, integrand, 50, volume_exact, surf_exact);
    }

    // q-convergence study for a 3D ellipsoid
    {
        auto ellipsoid = [](const uvector<real,3>& x)
        {
            return x(0)*x(0) + x(1)*x(1)*4 + x(2)*x(2)*9 - 1;
        };
        auto integrand = [](const uvector<real,3>& x)
        {
            return 1.0;
        };
        real volume_exact = (algoim::util::pi * 2) / 9;
        real surf_exact = 4.400809564664970341600200389229705943483674323377145800356686868037845;
        std::cout << "\n\nEllipsoid q-convergence test\n";
        std::cout << "q      volume(q)         surf(q)        vol error       surf error\n";
        qConv<3>(ellipsoid, -1.1, 1.1, 3, integrand, 50, volume_exact, surf_exact);
    }

    // Visusalisation of a 2D case involving a single polynomial; this example corresponds to
    // Figure 3, row 3, left column, https://doi.org/10.1016/j.jcp.2021.110720
    {
        auto phi = [](const uvector<real,2>& xx)
        {
            real x = xx(0)*2 - 1;
            real y = xx(1)*2 - 1;
            return -0.06225100787918392 + 0.1586472897571363*y + 0.5487135634635731*y*y + 
                x*(0.3478849533965025 - 0.3321074999999999*y - 0.5595163485848738*y*y) + 
                x*x*(0.7031095851739786 + 0.29459557349175747*y + 0.030425624999999998*y*y);
        };
        outputQuadScheme<2>(phi, 0.0, 1.0, 3, 3, "exampleA");
        std::cout << "\n\nQuadrature visualisation of a 2D case involving a single polynomial, corresponding\n";
        std::cout << "to Figure 3, row 3, left column, https://doi.org/10.1016/j.jcp.2021.110720, written\n";
        std::cout << "to exampleA-xxxx.vtp files (XML VTP file format).";
    }

    // Visusalisation of a 3D case involving a single polynomial; this example corresponds to
    // Figure 3, row 3, right column, https://doi.org/10.1016/j.jcp.2021.110720
    {
        auto phi = [](const uvector<real,3>& xx)
        {
            real x = xx(0)*2 - 1;
            real y = xx(1)*2 - 1;
            real z = xx(2)*2 - 1;
            return -0.3003521613375472 - 0.22416584292513722*z + 0.07904600284034838*z*z +
                y*(-0.022501556528537706 - 0.16299445153615613*z - 0.10968042065096766*z*z) + 
                y*y*(0.09321375574517882 - 0.07409794846221623*z + 0.09940785133211516*z*z) + 
                x*(0.094131400740032 - 0.11906280402685224*z - 0.010060302873268541*z*z + 
                y*y*(0.01448948481714108 - 0.0262370580373332*z - 0.08632912757566019*z*z) + 
                y*(0.08171132326327647 - 0.09286444275596013*z - 0.07651000354823911*z*z)) + 
                x*x*(-0.0914370528387867 + 0.09778971384044874*z - 0.1086777644685091*z*z + 
                y*y*(-0.04283439400630859 + 0.0750156999192893*z + 0.051754527934553866*z*z) + 
                y*(-0.052642188754328405 - 0.03538476045586772*z + 0.11117016852276898*z*z));
        };
        outputQuadScheme<3>(phi, 0.0, 1.0, 3, 3, "exampleB");
        std::cout << "\n\nQuadrature visualisation of a 3D case involving a single polynomial, corresponding\n";
        std::cout << "to Figure 3, row 3, right column, https://doi.org/10.1016/j.jcp.2021.110720, written\n";
        std::cout << "to exampleB-xxxx.vtp files (XML VTP file format).";
    }

    // Visusalisation of a 2D implicitly-defined domain involving the intersection of two polynomials; this example
    // corresponds to the top-left example of Figure 15, https://doi.org/10.1016/j.jcp.2021.110720
    {
        auto phi0 = [](const uvector<real,2>& xx)
        {
            real x = xx(0)*2 - 1;
            real y = xx(1)*2 - 1;
            return 0.014836540349115947 + 0.7022484024095262*y + 0.09974561176434385*y*y +
                x*(0.6863910464417281 + 0.03805619999999999*y - 0.09440658332756446*y*y) + 
                x*x*(0.19266932968830816 - 0.2325190091204104*y + 0.2957473125000001*y*y);
        };
        auto phi1 = [](const uvector<real,2>& xx)
        {
            real x = xx(0)*2 - 1;
            real y = xx(1)*2 - 1;
            return -0.18792528379702625 + 0.6713882473904913*y + 0.3778666084723582*y*y +
                x*x*(-0.14480813208127946 + 0.0897755603159206*y - 0.141199875*y*y) + 
                x*(-0.6169311810674598 - 0.19449299999999994*y - 0.005459163675646665*y*y);
        };
        outputQuadScheme<2>(phi0, phi1, 0.0, 1.0, 3, 3, "exampleC");
        std::cout << "\n\nQuadrature visualisation of a 2D implicitly-defined domain involving the\n";
        std::cout << "intersection of two polynomials, corresponding to the top-left example of Figure 15,\n";
        std::cout << "https://doi.org/10.1016/j.jcp.2021.110720, written to exampleC-xxxx.vtp files\n";
        std::cout << "(XML VTP file format).\n";
    }

    return 0;
}
#endif
