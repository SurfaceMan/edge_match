#include "edgesubpix3.h"

#include <float.h>

/* compute a > b considering the rounding errors due to the representation of double numbers*/
bool greater(float a, float b) {
    if (a <= b)
        return false; /* trivial case, return as soon as possible */
    if ((a - b) < 1000 * FLT_EPSILON)
        return false;
    return true; /* greater */
}

void compute_edge_points(cv::Mat &edge, const cv::Mat &modG, const cv::Mat &Gx, const cv::Mat &Gy) {
    int X = Gx.size().width;
    int Y = Gx.size().height;

    edge = cv::Mat(Gx.size(), CV_32FC2, {-1, -1});

/* explore pixels inside a 2 pixel margin (so modG[x,y +/- 1,1] is defined) */
#pragma omp parallel for
    for (int y = 2; y < (Y - 2); y++) {
        const auto *modPtr = modG.ptr<float>(y);
        const auto *gxPtr  = Gx.ptr<float>(y);
        const auto *gyPtr  = Gy.ptr<float>(y);

        for (int x = 2; x < (X - 2); x++) {
            int  Dx  = 0; /* interpolation is along Dx,Dy		*/
            int  Dy  = 0; /* which will be selected below		*/
            auto mod = modPtr[ x ];
            auto L   = modPtr[ x - 1 ];
            auto R   = modPtr[ x + 1 ];
            auto U   = modPtr[ x + X ];
            auto D   = modPtr[ x - X ];
            auto gx  = fabs(gxPtr[ x ]);
            auto gy  = fabs(gyPtr[ x ]);
            // auto mod = modG.at<float>({x, y});       /* modG at pixel					*/
            // auto L   = modG.at<float>({x - 1, y});   /* modG at pixel on the left			*/
            // auto R   = modG.at<float>({x + 1, y});   /* modG at pixel on the right		*/
            // auto U   = modG.at<float>({x, (y + 1)}); /* modG at pixel up					*/
            // auto D   = modG.at<float>({x, (y - 1)}); /* modG at pixel below				*/
            // auto gx  = fabs(Gx.at<float>({x, y}));   /* absolute value of Gx				*/
            // auto gy  = fabs(Gy.at<float>({x, y}));   /* absolute value of Gy				*/
            /* when local horizontal maxima of the gradient modulus and the gradient direction
            is more horizontal (|Gx| >= |Gy|),=> a "horizontal" (H) edge found else,
            if local vertical maxima of the gradient modulus and the gradient direction is more
            vertical (|Gx| <= |Gy|),=> a "vertical" (V) edge found */

            /* it can happen that two neighbor pixels have equal value and are both	maxima, for
            example when the edge is exactly between both pixels. in such cases, as an arbitrary
            convention, the edge is marked on the left one when an horizontal max or below when a
            vertical max. for	this the conditions are L < mod >= R and D < mod >= U,respectively.
            the comparisons are done using the function greater() instead of the operators > or >=
            so numbers differing only due to rounding errors are considered equal */
            if (greater(mod, L) && !greater(R, mod) && gx >= gy)
                Dx = 1; /* H */
            else if (greater(mod, D) && !greater(U, mod) && gx <= gy)
                Dy = 1; /* V */

            /* Devernay sub-pixel correction

             the edge point position is selected as the one of the maximum of a quadratic
             interpolation of the magnitude of the gradient along a unidimensional direction. the
             pixel must be a local maximum. so we	have the values:

              the x position of the maximum of the parabola passing through(-1,a), (0,b), and (1,c)
             is offset = (a - c) / 2(a - 2b + c),and because b >= a and b >= c, -0.5 <= offset <=
             0.5
              */
            if (Dx == 0 && Dy == 0) {
                continue;
            }

            /* offset value is in [-0.5, 0.5] */
            float a;
            float b = mod;
            float c;
            if (Dx > 0) {
                a = L;
                c = R;
            } else if (Dy > 0) {
                a = D;
                c = U;
            }
            auto offset = 0.5f * (a - c) / (a - b - b + c);

            /* store edge point */
            edge.at<cv::Vec2f>({x, y}) = {x + offset * Dx, y + offset * Dy};
        }
    }
}

/* return a score for chaining pixels 'from' to 'to', favoring closet point:
    = 0.0 invalid chaining;
    > 0.0 valid forward chaining; the larger the value, the better the chaining;
    < 0.0 valid backward chaining; the smaller the value, the better the chaining;

input:
 from, to: the two pixel IDs to evaluate their potential chaining
 Ex[i], Ey[i]: the sub-pixel position of point i, if i is an edge point;
 they take values -1,-1 if i is not an edge point;
 Gx[i], Gy[i]: the image gradient at pixel i;
 X, Y: the size of the image;
*/
float chain(const cv::Point &from, const cv::Point &to, const cv::Mat &edge, const cv::Mat &Gx,
            const cv::Mat &Gy) {
    // check that the points are different and valid edge points,otherwise return invalid chaining
    if (from == to)
        return 0.0f; // same pixel, not a valid chaining

    auto &edgeFrom = edge.at<cv::Vec2f>(from);
    auto &edgeTo   = edge.at<cv::Vec2f>(to);

    auto gradXFrom = Gx.at<float>(from);
    auto gradXTo   = Gx.at<float>(to);
    auto gradYFrom = Gy.at<float>(from);
    auto gradYTo   = Gy.at<float>(to);

    if (edgeFrom[ 0 ] < 0.0f || edgeTo[ 0 ] < 0.0f || edgeFrom[ 1 ] < 0.0f || edgeTo[ 1 ] < 0.0f)
        return 0.0f; // one of them is not an edge point, not a valid chaining

    /* in a good chaining, the gradient should be roughly orthogonal
    to the line joining the two points to be chained:
    when Gy * dx - Gx * dy > 0, it corresponds to a forward chaining,
    when Gy * dx - Gx * dy < 0, it corresponds to a backward chaining.

     first check that the gradient at both points to be chained agree
     in one direction, otherwise return invalid chaining. */
    auto dx          = edgeTo[ 0 ] - edgeFrom[ 0 ];
    auto dy          = edgeTo[ 1 ] - edgeFrom[ 1 ];
    auto fromProject = gradYFrom * dx - gradXFrom * dy;
    auto toProject   = gradYTo * dx - gradXTo * dy;
    if (fromProject * toProject <= 0.0f)
        return 0.0f; /* incompatible gradient angles, not a valid chaining */

    /* return the chaining score: positive for forward chaining,negative for backwards.
    the score is the inverse of the distance to the chaining point, to give preference to closer
    points */
    auto dist = sqrt(dx * dx + dy * dy);
    if (fromProject >= 0.0)
        return 1.0f / dist; /* forward chaining  */
    else
        return -1.0f / dist; /* backward chaining */
}

/* chain edge points
input:
Ex/Ey:the sub-pixel coordinates when an edge point is present or -1,-1 otherwise.
Gx/Gy/modG:the x and y components and the modulus of the image gradient. X,Y is the image size.

output:
next and prev:contain the number of next and previous edge points in the chain.
when not chained in one of the directions, the corresponding value is set to -1.
next and prev must be allocated before calling.*/
void chain_edge_points(cv::Mat &next, cv::Mat &prev, const cv::Mat &edge, const cv::Mat &Gx,
                       const cv::Mat &Gy) {
    int X = edge.size().width;
    int Y = edge.size().height;

    next = cv::Mat(edge.size(), CV_32SC2, {-1, -1});
    prev = cv::Mat(edge.size(), CV_32SC2, {-1, -1});

/* try each point to make local chains */
#pragma omp parallel for
    for (int y = 2; y < (Y - 2); y++) { /* 2 pixel margin to include the tested neighbors */
        for (int x = 2; x < (X - 2); x++) {
            cv::Point from(x, y); /* edge point to be chained			*/
            /* must be an edge point */
            auto &edgeP = edge.at<cv::Vec2f>(from);
            if (edgeP[ 0 ] < 0.0f || edgeP[ 1 ] < 0.0f) {
                continue;
            }

            float     forwardScore  = 0.0;      /* score of best forward chaining		*/
            float     backwardScore = 0.0;      /* score of best backward chaining		*/
            cv::Point forwardPos    = {-1, -1}; /* edge point of best forward chaining */
            cv::Point backwardPos   = {-1, -1}; /* edge point of best backward chaining*/

            /* try all neighbors two pixels apart or less.
            looking for candidates for chaining two pixels apart, in most such cases,
            is enough to obtain good chains of edge points that	accurately describes the edge.
            */
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    cv::Point to(x + i, y + j); /* candidate edge point to be chained */
                    auto      score = chain(from, to, edge, Gx, Gy); /* score from-to */

                    if (score > forwardScore) /* a better forward chaining found    */
                    {
                        forwardScore = score; /* set the new best forward chaining  */
                        forwardPos   = to;
                    }
                    if (score < backwardScore) /* a better backward chaining found	  */
                    {
                        backwardScore = score; /* set the new best backward chaining */
                        backwardPos   = to;
                    }
                }
            }
            /* before making the new chain, check whether the target was
            already chained and in that case, whether the alternative
            chaining is better than the proposed one.

             x alt                        x alt
             \                          /
             \                        /
             from x---------x fwd              bck x---------x from

              we know that the best forward chain starting at from is from-fwd.
              but it is possible that there is an alternative chaining arriving
              at fwd that is better, such that alt-fwd is to be preferred to
              from-fwd. an analogous situation is possible in backward chaining,
              where an alternative link bck-alt may be better than bck-from.

               before making the new link, check if fwd/bck are already chained,
               and in such case compare the scores of the proposed chaining to
               the existing one, and keep only the best of the two.

                there is an undesirable aspect of this procedure: the result may
                depend on the order of exploration. consider the following
                configuration:

                 a x-------x b
                 /
                 /
                 c x---x d    with score(a-b) < score(c-b) < score(c-d)
                 or equivalently ||a-b|| > ||b-c|| > ||c-d||

                  let us consider two possible orders of exploration.

                   order: a,b,c
                   we will first chain a-b when exploring a. when analyzing the
                   backward links of b, we will prefer c-b, and a-b will be unlinked.
                   finally, when exploring c, c-d will be preferred and c-b will be
                   unlinked. the result is just the chaining c-d.

                    order: c,b,a
                    we will first chain c-d when exploring c. then, when exploring
                    the backward connections of b, c-b will be the preferred link;
                    but because c-d exists already and has a better score, c-b
                    cannot be linked. finally, when exploring a, the link a-b will
                    be created because there is no better backward linking of b.
                    the result is two chainings: c-d and a-b.

                     we did not found yet a simple algorithm to solve this problem. by
                     simple, we mean an algorithm without two passes or the need to
                     re-evaluate the chaining of points where one link is cut.

                      for most edge points, there is only one possible chaining and this
                      problem does not arise. but it does happen and a better solution
                      is desirable.
                      */

            auto     &nextForm = next.at<cv::Point>(from);
            cv::Point preForward;
            if (forwardPos.x >= 0 && nextForm != forwardPos &&
                ((preForward = prev.at<cv::Point>(forwardPos)).x < 0 ||
                 chain(preForward, forwardPos, edge, Gx, Gy) < forwardScore)) {

                /* remove previous from-x link if one */
                /* only prev requires explicit reset  */
                /* set next of from-fwd link          */
                if (nextForm.x >= 0)
                    prev.at<cv::Point>(nextForm) = {-1, -1};
                nextForm = forwardPos;

                /* remove alt-fwd link if one         */
                /* only next requires explicit reset  */
                /* set prev of from-fwd link          */
                if (preForward.x >= 0)
                    next.at<cv::Point>(preForward) = {-1, -1};
                prev.at<cv::Point>(forwardPos) = from;
            }

            cv::Point nextBackward;
            auto     &preFrom = prev.at<cv::Point>(from);
            if (backwardPos.x >= 0 && preFrom != backwardPos &&
                ((nextBackward = next.at<cv::Point>(backwardPos)).x < 0 ||
                 chain(nextBackward, backwardPos, edge, Gx, Gy) > backwardScore)) {

                /* remove bck-alt link if one         */
                /* only prev requires explicit reset  */
                /* set next of bck-from link          */
                if (nextBackward.x >= 0)
                    prev.at<cv::Point>(nextBackward) = {-1, -1};
                next.at<cv::Point>(backwardPos) = from;

                /* remove previous x-from link if one */
                /* only next requires explicit reset  */
                /* set prev of bck-from link          */
                if (preFrom.x >= 0)
                    next.at<cv::Point>(preFrom) = {-1, -1};
                preFrom = backwardPos;
            }
        }
    }
}

/* apply Canny thresholding with hysteresis

next and prev contain the number of next and previous edge points in the
chain or -1 when not chained. modG is modulus of the image gradient. X,Y is
the image size. th_h and th_l are the high and low thresholds, respectively.

this function modifies next and prev, removing chains not satisfying the
thresholds.
*/
void thresholds_with_hysteresis(cv::Mat &next, cv::Mat &prev, const cv::Mat &modG, float high,
                                float low) {

    cv::Mat valid(modG.size(), CV_8UC1, cv::Scalar(0));

    int X = modG.size().width;
    int Y = modG.size().height;

/* validate all edge points over th_h or connected to them and over th_l */
#pragma omp parallel for
    for (int row = 0; row < Y; row++) { /* prev[i]>=0 or next[i]>=0 implies an edge point */
        for (int col = 0; col < X; col++) {
            cv::Point pos{col, row};
            if ((prev.at<cv::Point>(pos).x >= 0 || next.at<cv::Point>(pos).x >= 0) &&
                !valid.at<bool>(pos) && modG.at<float>(pos) >= high) {
                valid.at<bool>(pos) = true; /* mark as valid the new point */

                /* follow the chain of edge points forwards */
                cv::Point nextPos = pos;
                cv::Point tmp;
                for (; nextPos.x >= 0 && (tmp = next.at<cv::Point>(nextPos)).x >= 0 &&
                       !valid.at<bool>(tmp);
                     nextPos = next.at<cv::Point>(nextPos)) {
                    if (modG.at<float>(tmp) < low) {
                        /* cut the chain when the point is below th_l */
                        next.at<cv::Point>(nextPos) = {-1, -1};
                        /* j must be assigned to next[j] and not k, so the loop is chained in this
                         * case */
                        prev.at<cv::Point>(tmp) = {-1, -1};

                    } else {
                        valid.at<bool>(tmp) = true; /* otherwise mark the new point as valid */
                    }
                }

                /* follow the chain of edge points backwards */
                cv::Point prePos = pos;
                for (; prePos.x >= 0 && (tmp = prev.at<cv::Point>(prePos)).x >= 0 &&
                       !valid.at<bool>(tmp);
                     prePos = prev.at<cv::Point>(prePos)) {
                    if (modG.at<float>(tmp) < low) {
                        /* cut the chain when the point is below th_l */
                        prev.at<cv::Point>(prePos) = {-1, -1};
                        /* j must be assigned to prev[j] and not k, so the loop is chained in this
                         * case */
                        next.at<cv::Point>(tmp) = {-1, -1};

                    } else {
                        valid.at<bool>(tmp) = true; /* otherwise mark the new point as valid */
                    }
                }
            }
        }
    }

/* remove any remaining non-valid chained point */
#pragma omp parallel for
    for (int row = 0; row < Y; row++) { /* prev[i]>=0 or next[i]>=0 implies an edge point */
        for (int col = 0; col < X; col++) {
            cv::Point pos{col, row};
            auto     &prePos  = prev.at<cv::Point>(pos);
            auto     &nextPos = next.at<cv::Point>(pos);
            if ((prePos.x > 0 || nextPos.x > 0) && !valid.at<bool>(pos)) {
                prePos = nextPos = {-1, -1};
            }
        }
    }
}

/* create a list of chained edge points composed of 3 lists
x, y and curve_limits; it also computes N (the number of edge points) and
M (the number of curves).

x[i] and y[i] (0<=i<N) store the sub-pixel coordinates of the edge points.
curve_limits[j] (0<=j<=M) stores the limits of each chain in lists x and y.

x, y, and curve_limits will be allocated.

example:

curve number k (0<=k<M) consists of the edge points x[i],y[i]
for i determined by curve_limits[k] <= i < curve_limits[k+1].

curve k is closed if x[curve_limits[k]] == x[curve_limits[k+1] - 1] and
y[curve_limits[k]] == y[curve_limits[k+1] - 1].
*/
void list_chained_edge_points(std::vector<std::vector<cv::Point2f>> &points,
                              std::vector<std::vector<cv::Vec2f>> &dirs, cv::Mat &next,
                              cv::Mat &prev, const cv::Mat &edge, const cv::Mat &Gx,
                              const cv::Mat &Gy, const cv::Mat &mag) {

    /* initialize output: x, y, curve_limits, N, and M

     there cannot be more than X*Y edge points to be put in the output list:
     edge points must be local maxima of gradient modulus, so at most half of
     the pixels could be so. when a closed curve is found, one edge point will
     be put twice to the output. even if all possible edge points (half of the
     pixels in the image) would form one pixel closed curves (which is not
     possible) that would lead to output X*Y edge points.

      for the same reason, there cannot be more than X*Y curves: the worst case
      is when all possible edge points (half of the pixels in the image) would
      form one pixel chains. in that case (which is not possible) one would need
      a size for curve_limits of X*Y/2+1. so X*Y is enough.

     (curve_limits requires one more item than the number of curves.
     a simplest example is when only one chain of length 3 is present:
     curve_limits[0] = 0, curve_limits[1] = 3.)
     */

    /* copy chained edge points to output */

    int X = edge.size().width;
    int Y = edge.size().height;
    points.clear();
    dirs.clear();

    /* prev[i]>=0 or next[i]>=0 implies an edge point */
    for (int row = 0; row < Y; row++) {
        for (int col = 0; col < X; col++) {
            cv::Point pos{col, row};

            if (prev.at<cv::Point>(pos).x >= 0 || next.at<cv::Point>(pos).x >= 0) {
                /* a new chain found, set chain starting index to the current point
                and then increase the curve counter */
                std::vector<cv::Point2f> path;
                std::vector<cv::Vec2f>   dir;

                cv::Point k = pos;
                cv::Point n;
                /* set k to the beginning of the chain, or to i if closed curve */
                for (; (n = prev.at<cv::Point>(k)).x >= 0 && n != pos; k = n)
                    ;

                /* follow the chain of edge points starting on k */
                do {
                    /* store the current point coordinates in the output lists */
                    auto mog = mag.at<float>(k);
                    path.emplace_back(edge.at<cv::Vec2f>(k));
                    dir.emplace_back(Gx.at<float>(k) / mog, Gy.at<float>(k) / mog);

                    n = next.at<cv::Point>(k); /* save the id of the next point in the chain */
                    /* unlink chains from k so it is not used again */
                    next.at<cv::Point>(k) = {-1, -1};
                    prev.at<cv::Point>(k) = {-1, -1};

                    /* for closed curves, the initial point is included again as
                    the last point of the chain. actually, testing if the first
                    and last points are equal is the only way to know that it is
                    a closed curve.

                     to understand that this code actually repeats the first point,
                     consider a closed chain as follows:  a--b
                     |  |
                     d--c

                      let us say that the algorithm starts by point a. it will store
                      the coordinates of point a and then unlink a-b. then, will store
                      point b and unlink b-c, and so on. but the link d-a is still
                      there. (point a is no longer pointing backwards to d, because
                      both links are removed at each step. but d is indeed still
                      pointing to a.) so it will arrive at point a again and store its
                      coordinates again as last point. there, it cannot continue
                      because the link a-b was removed, there would be no next point,
                      k would be -1 and the curve is finished.
                      */

                    k = n;          /* set the current point to the next in the chain */
                } while (k.x >= 0); /* continue while there is a next point in the chain */

                points.emplace_back(std::move(path));
                dirs.emplace_back(std::move(dir));
            }
        }
    }
}

void EdgePoint3(const cv::Mat &img, std::vector<std::vector<cv::Point2f>> &points,
                std::vector<std::vector<cv::Vec2f>> &dirs, float sigma, float low, float high) {
    auto    start = cv::getTickCount();
    cv::Mat blured;
    cv::GaussianBlur(img, blured, cv::Size(5, 5), 0);

    cv::Mat dx;
    cv::Mat dy;
    cv::Mat mag;
    cv::spatialGradient(blured, dx, dy);

    cv::Mat dx2;
    cv::Mat dy2;
    dx.convertTo(dx2, CV_32F, 0.25);
    dy.convertTo(dy2, CV_32F, 0.25);
    cv::magnitude(dx2, dy2, mag);
    {
        auto  end   = cv::getTickCount();
        int64 delta = end - start;
        auto  time  = (delta / cv::getTickFrequency()) * 1000; // units msec
        std::cout << "pre cost(ms):" << time << std::endl;
    }

    // inter
    cv::Mat edge;
    auto    start2 = cv::getTickCount();

    compute_edge_points(edge, mag, dx2, dy2);

    {
        auto  end   = cv::getTickCount();
        int64 delta = end - start2;
        auto  time  = (delta / cv::getTickFrequency()) * 1000; // units msec
        std::cout << "compute_edge_points cost(ms):" << time << std::endl;
    }

    cv::Mat next;
    cv::Mat prev;

    auto start3 = cv::getTickCount();
    chain_edge_points(next, prev, edge, dx2, dy2);

    {
        auto  end   = cv::getTickCount();
        int64 delta = end - start3;
        auto  time  = (delta / cv::getTickFrequency()) * 1000; // units msec
        std::cout << "chain_edge_points cost(ms):" << time << std::endl;
    }

    auto start4 = cv::getTickCount();
    thresholds_with_hysteresis(next, prev, mag, high, low);

    {
        auto  end   = cv::getTickCount();
        int64 delta = end - start4;
        auto  time  = (delta / cv::getTickFrequency()) * 1000; // units msec
        std::cout << "thresholds_with_hysteresis cost(ms):" << time << std::endl;
    }

    auto start5 = cv::getTickCount();
    list_chained_edge_points(points, dirs, next, prev, edge, dx2, dy2, mag);

    {
        auto  end   = cv::getTickCount();
        int64 delta = end - start5;
        auto  time  = (delta / cv::getTickFrequency()) * 1000; // units msec
        std::cout << "list_chained_edge_points cost(ms):" << time << std::endl;
    }
}
