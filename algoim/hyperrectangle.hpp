#ifndef ALGOIM_HYPERRECTANGLE_HPP
#define ALGOIM_HYPERRECTANGLE_HPP

// algoim::HyperRectangle<T,N>

#include "real.hpp"
#include "uvector.hpp"

namespace algoim
{
    // HyperRectangle<T,N> describes the extent of a hyperrectangle, i.e., the set of points
    // x = (x(0), ..., x(i), ..., x(N-1)) such that xmin(i) <= x(i) <= xmax(i) for all i.
    template<typename T, int N>
    struct HyperRectangle
    {
        uvector<uvector<T,N>,2> range;

        HyperRectangle(const uvector<T,N>& min, const uvector<T,N>& max) : range{min, max} {}

        const uvector<T,N>& side(int s) const
        {
            return range(s);
        }

        const uvector<T,N>& min() const
        {
            return range(0);
        }

        T& min(int i)
        {
            return range(0)(i);
        }

        T min(int i) const
        {
            return range(0)(i);
        }

        const uvector<T,N>& max() const
        {
            return range(1);
        }

        T& max(int i)
        {
            return range(1)(i);
        }

        T max(int i) const
        {
            return range(1)(i);
        }

        uvector<T,N> extent() const
        {
            return range(1) - range(0);
        }

        T extent(int i) const
        {
            return range(1)(i) - range(0)(i);
        }

        uvector<real,N> midpoint() const
        {
            return (range(0) + range(1)) * 0.5;
        }

        real midpoint(int i) const
        {
            return (range(0)(i) + range(1)(i)) * 0.5;
        }

        bool operator==(const HyperRectangle& x) const
        {
            for (int dim = 0; dim < N; ++dim)
                if (range(0)(dim) != x.range(0)(dim) || range(1)(dim) != x.range(1)(dim))
                    return false;
            return true;
        }
    };
} // namespace algoim

#endif
