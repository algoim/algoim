#ifndef ALGOIM_KDTREE_HPP
#define ALGOIM_KDTREE_HPP

/* algoim::KDTree<T,N> constructs a k-d tree data structure for a given collection of
    points of type uvector<T,N>, where T is typically float or double, and N is 
    the dimension. This particular implementation of the k-d tree has been optimised for the 
    case that the points are situated on a smooth codimension-one surface by using coordinate
    transformations that result in "tight" bounding boxes for some of the nodes in the tree.
       
    Refer to the paper
        R. I. Saye, High-order methods for computing distances to implicitly defined surfaces,
        Communications in Applied Mathematics and Computational Science, 9(1), 107-141 (2014),
        http://dx.doi.org/10.2140/camcos.2014.9.107
    for more information. */

#include <vector>
#include <cassert>
#include "real.hpp"
#include "uvector.hpp"
#include "utility.hpp"

namespace algoim
{
    namespace detail
    {
        // Returns the squared-distance from a point x to the hyperrectangle [min,max] (if x is inside the rectangle, the distance is zero)
        template<typename T, int N>
        T sqrDistBox(const uvector<T,N>& x, const uvector<T,N>& min, const uvector<T,N>& max)
        {
            T dsqr = T(0);
            for (int i = 0; i < N; ++i)
                if (x(i) < min(i))
                    dsqr += util::sqr(x(i) - min(i));
                else if (x(i) > max(i))
                    dsqr += util::sqr(x(i) - max(i));
            return dsqr;
        }
    }

    template<typename T, int N>
    class KDTree
    {
        std::vector<uvector<T,N>> points;
        std::vector<int> index;
        static constexpr int leaf_size = 16;

        // A Node of the tree is a leaf node iff type=-1. If type >= 0, then a coordinate transform
        // is applied. If type < -1, the node is a standard splitting node with two children and no transform
        struct Node
        {
            int type;
            int i0, i1;
            uvector<T,N> xmin, xmax;
        };

        std::vector<Node> nodes;
        std::vector<uvector<uvector<T,N>,N>> transforms;

        // Given a node and a range of points [lb,ub), build the tree
        void build_tree(int nodeIndex, int lb, int ub, bool hasTransformed, int level)
        {
            assert(lb < ub);
            Node& node = nodes[nodeIndex];

            // Compute bounding box and mean
            uvector<T,N> mean = node.xmin = node.xmax = points[index[lb]];
            for (int i = lb + 1; i < ub; ++i)
            {
                const uvector<T,N>& x = points[index[i]];
                mean += x;
                for (int j = 0; j < N; ++j)
                {
                    if (x(j) < node.xmin(j)) node.xmin(j) = x(j);
                    if (x(j) > node.xmax(j)) node.xmax(j) = x(j);
                }
            }
            mean /= static_cast<T>(ub - lb);

            // The node is a leaf iff point count is sufficiently small
            if (ub - lb <= leaf_size)
            {
                node.type = -1;
                node.i0 = lb;
                node.i1 = ub;
                return;
            }

            // Splitting node: default to splitting along greatest extent
            node.type = -2;
            int axis = argmax(node.xmax - node.xmin);

            // Evaluate possibility for coordinate transformation
            if (!hasTransformed && level > 5 && ub - lb >= leaf_size * (1 << 2))
            {
                // Estimate normal
                T holeRadiusSqr = util::sqr(0.05*max(node.xmax - node.xmin));
                uvector<T,N> n = static_cast<T>(0);
                n(0) = 1.0;
                for (int i = lb; i < ub; ++i)
                {
                    uvector<T,N> x = points[index[i]] - mean;
                    T msqr = sqrnorm(x);
                    if (msqr > holeRadiusSqr)
                        n -= x * (dot(x,n)/msqr);
                }
                T msqr = sqrnorm(n);
                if (msqr == 0.0)
                    n(0) = 1.0;
                else
                    n /= sqrt(msqr);

                // Compute alpha range
                T minAlpha = std::numeric_limits<T>::max();
                T maxAlpha = -std::numeric_limits<T>::max();
                for (int i = lb; i < ub; ++i)
                {
                    T alpha = dot(points[index[i]], n);
                    if (alpha > maxAlpha) maxAlpha = alpha;
                    if (alpha < minAlpha) minAlpha = alpha;
                }

                if (maxAlpha - minAlpha < 0.1*max(node.xmax - node.xmin))
                {
                    // Perform transformation: calculate an orthonormal basis using the normal as first axis.
                    // A stable method for doing so is to compute the Householder matrix which maps n to ej,
                    // i.e., P = I - 2 uu^T where u = normalised(n - ej), where ej is the j-th basis vector,
                    // j chosen that that n != ej.
                    uvector<uvector<T,N>,N> axes;
                    int j = argmin(abs(n));
                    uvector<T,N> u = n; u(j) -= 1.0;
                    u /= norm(u);
                    for (int dim = 0; dim < N; ++dim)
                        for (int i = 0; i < N; ++i)
                            axes(dim)(i) = (dim == i ? 1.0 : 0.0) - 2.0 * u(dim) * u(i);

                    // Swap the first row of axes with j, so that the normal is the first axis. This is likely
                    // unnecessary (but done in the name of consistency with old approach).
                    std::swap(axes(0), axes(j));

                    // Apply coordinate transformation and calculate new bounding box in order to determine new split direction
                    uvector<T,N> bmin = std::numeric_limits<T>::max();
                    uvector<T,N> bmax = -std::numeric_limits<T>::max();
                    for (int i = lb; i < ub; ++i)
                    {
                        uvector<T,N> x = points[index[i]];
                        for (int dim = 0; dim < N; ++dim)
                        {
                            T alpha = dot(axes(dim), x);
                            points[index[i]](dim) = alpha;
                            if (alpha < bmin(dim)) bmin(dim) = alpha;
                            if (alpha > bmax(dim)) bmax(dim) = alpha;                    
                        }
                    }

                    node.type = static_cast<int>(transforms.size());
                    transforms.push_back(axes);
                    axis = argmax(bmax - bmin);
                    hasTransformed = true;
                }
            }

            // Use median as the split
            int m = (lb + ub)/2;

            // Rearrange points
            std::nth_element(index.begin() + lb, index.begin() + m, index.begin() + ub, [&](int i0, int i1) { return points[i0](axis) < points[i1](axis); } );

            // Build child trees
            int i0 = node.i0 = static_cast<int>(nodes.size());
            int i1 = node.i1 = static_cast<int>(nodes.size() + 1);
            nodes.push_back(Node());
            nodes.push_back(Node());
            build_tree(i0, lb, m, hasTransformed, level + 1);
            build_tree(i1, m, ub, hasTransformed, level + 1);
        }

        struct ClosestPoint
        {
            uvector<T,N> x;
            T distsqr;
            int ind;
        };

        // Recursive function for searching the tree for the closest point to a given point cp.x
        void search(const Node& node, ClosestPoint& cp) const
        {
            if (node.type == -1)
            {
                // Leaf node
                for (int j = node.i0; j < node.i1; ++j)
                {
                    T dsqr = sqrnorm(points[j] - cp.x);
                    if (dsqr < cp.distsqr)
                    {
                        cp.distsqr = dsqr;
                        cp.ind = j;
                    }
                }
            }
            else
            {
                // Non-leaf node
                if (node.type >= 0)
                {
                    // Transform query point to new coordinate system
                    const uvector<uvector<T,N>,N>& axes = transforms[node.type];
                    uvector<T,N> x = cp.x;
                    for (int dim = 0; dim < N; ++dim)
                        cp.x(dim) = dot(axes(dim), x);
                }

                T dleft = detail::sqrDistBox(cp.x, nodes[node.i0].xmin, nodes[node.i0].xmax);
                T dright = detail::sqrDistBox(cp.x, nodes[node.i1].xmin, nodes[node.i1].xmax);
                if (dleft < dright)
                {
                    if (dleft < cp.distsqr)
                    {
                        search(nodes[node.i0], cp);
                        if (dright < cp.distsqr)
                            search(nodes[node.i1], cp);
                    }
                }
                else
                {
                    if (dright < cp.distsqr)
                    {
                        search(nodes[node.i1], cp);
                        if (dleft < cp.distsqr)
                            search(nodes[node.i0], cp);
                    }
                }

                if (node.type >= 0)
                {
                    // Transform query point back to old coordinate system. This is about 5% faster than storing
                    // the old value of cp.x on the stack
                    const uvector<uvector<T,N>,N>& axes = transforms[node.type];
                    uvector<T,N> x = cp.x;
                    cp.x = axes(0)*x(0);
                    for (int dim = 1; dim < N; ++dim)
                        cp.x += axes(dim)*x(dim);
                }
            }
        }

    public:

        // Construct a KDTree from a given collection of points p (the given points are not overwritten)
        KDTree(const std::vector<uvector<T,N>>& p)
        {
            assert(p.size() < std::numeric_limits<int>::max()); // code currently uses int to index but could easily be adapted to size_t
            int len = static_cast<int>(p.size());
            if (len == 0)
                return;

            // Copy points
            points = p;

            // Build initial index array, which shall soon be reordered
            index.resize(len);
            for (int i = 0; i < len; ++i)
                index[i] = i;

            // Recursively build tree starting from root node. This only manipulates index array
            nodes.push_back(Node());
            build_tree(0, 0, len, false, 0);

            // Rearrange so that points in leaf nodes are contiguous in memory. Based on tree
            // construction, both nodes & point ranges will be organised depth-first.
            std::vector<uvector<T,N>> pointscopy(points);
            for (size_t i = 0; i < len; ++i)
                points[i] = pointscopy[index[i]];
        }

        // Search the tree for the closest point to x, where the search is restricted to all points that are
        // within a squared distance of rsqr to x. Returns -1 if no such point exists, otherwise the index
        // of the closest point in the original array p passed to the KDTree constructor is returned.
        int nearest(const uvector<T,N>& x, T rsqr = std::numeric_limits<T>::max()) const
        {
            if (nodes.empty())
                return -1;
            ClosestPoint cp;
            cp.x = x;
            cp.distsqr = rsqr;
            cp.ind = -1;
            search(nodes[0], cp);
            return cp.ind >= 0 ? index[cp.ind] : -1;
        }
    };
} // namespace algoim

#endif
