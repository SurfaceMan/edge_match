#include "edgesubpix.h"
#include "gradient.h"

void compute_edge_points(cv::Mat &edge, const cv::Mat &mag, const cv::Mat &grad, float low) {
    const int X  = grad.size().width;
    const int Y  = grad.size().height;
    low         *= low;

    /* explore pixels inside a 2 pixel margin (so modG[x,y +/- 1,1] is defined) */
    for (int y = 2; y < (Y - 2); y++) {
        for (int x = 2; x < (X - 2); x++) {
            const auto mod = mag.at<unsigned short>({x, y}); /* modG at pixel					*/
            if (mod < low) {
                continue;
            }

            const auto L = mag.at<unsigned short>({x - 1, y});   /* modG at pixel on the left	*/
            const auto R = mag.at<unsigned short>({x + 1, y});   /* modG at pixel on the right	*/
            const auto U = mag.at<unsigned short>({x, (y + 1)}); /* modG at pixel up			*/
            const auto D = mag.at<unsigned short>({x, (y - 1)}); /* modG at pixel below			*/

            const auto &gradVal = grad.at<cv::Vec2s>({x, y});
            const auto  gx      = abs(gradVal[ 0 ]); /* absolute value of Gx			*/
            const auto  gy      = abs(gradVal[ 1 ]); /* absolute value of Gy			*/
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

            int Dx = 0; /* interpolation is along Dx,Dy		*/
            int Dy = 0; /* which will be selected below		*/
            if ((mod > L) && !(R > mod) && gx >= gy) {
                Dx = 1; /* H */
            } else if ((mod > D) && !(U > mod) && gx <= gy) {
                Dy = 1; /* V */
            }
            /* Devernay sub-pixel correction

             the edge point position is selected as the one of the maximum of a quadratic
             interpolation of the magnitude of the gradient along a unidimensional direction. the
             pixel must be a local maximum. so we	have the values:

              the x position of the maximum of the parabola passing through(-1,a), (0,b), and (1,c)
             is offset = (a - c) / 2(a - 2b + c),and because b >= a and b >= c, -0.5 <= offset <=
             0.5
              */
            if (Dx > 0) {
                const float a      = sqrtf(D);
                const float b      = sqrtf(mod);
                const float c      = sqrtf(U);
                const float offset = 0.5f * (a - c) / (a - b - b + c);

                /* store edge point */
                edge.at<cv::Point2f>({x, y}) = {x + offset * Dx, y + offset * Dy};

            } else if (Dy > 0) {
                const float a      = sqrtf(R);
                const float b      = sqrtf(mod);
                const float c      = sqrtf(L);
                const float offset = 0.5f * (a - c) / (a - b - b + c);

                /* store edge point */
                edge.at<cv::Point2f>({x, y}) = {x + offset * Dx, y + offset * Dy};
            }
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
float chain(const cv::Point &from, const cv::Point &to, const cv::Mat &edge, const cv::Mat &grad) {
    // check that the points are different and valid edge points,otherwise return invalid chaining
    if (from == to) {
        return 0.0; // same pixel, not a valid chaining
    }
    const auto edgeFrom = edge.at<cv::Point2f>(from);
    const auto edgeTo   = edge.at<cv::Point2f>(to);
    if (edgeFrom.x < 0.0 || edgeTo.x < 0.0 || edgeFrom.y < 0.0 || edgeTo.y < 0.0) {
        return 0.0; // one of them is not an edge point, not a valid chaining
    }

    const auto gradFrom = grad.at<cv::Vec2s>(from);
    const auto gradTo   = grad.at<cv::Vec2s>(to);

    /* in a good chaining, the gradient should be roughly orthogonal
    to the line joining the two points to be chained:
    when Gy * dx - Gx * dy > 0, it corresponds to a forward chaining,
    when Gy * dx - Gx * dy < 0, it corresponds to a backward chaining.

     first check that the gradient at both points to be chained agree
     in one direction, otherwise return invalid chaining. */
    const auto delta       = edgeTo - edgeFrom;
    const auto fromProject = gradFrom[ 1 ] * delta.x - gradFrom[ 0 ] * delta.y;
    const auto toProject   = gradTo[ 1 ] * delta.x - gradTo[ 0 ] * delta.y;
    if (fromProject * toProject <= 0.0) {
        return 0.0; /* incompatible gradient angles, not a valid chaining */
    }

    /* return the chaining score: positive for forward chaining,negative for backwards.
    the score is the inverse of the distance to the chaining point, to give preference to closer
    points */
    const auto dist = cv::norm(delta);
    if (fromProject >= 0.0) {
        return 1.0f / dist; /* forward chaining  */
    }
    return -1.0f / dist; /* backward chaining */
}

float chain(const cv::Point2f &edgeFrom,
            const cv::Vec2s   &gradFrom,
            const cv::Point   &to,
            const cv::Mat     &edge,
            const cv::Mat     &grad) {
    const auto edgeTo = edge.at<cv::Point2f>(to);
    if (edgeTo.x < 0.0) {
        return 0.0; // one of them is not an edge point, not a valid chaining
    }

    const auto gradTo = grad.at<cv::Vec2s>(to);

    /* in a good chaining, the gradient should be roughly orthogonal
    to the line joining the two points to be chained:
    when Gy * dx - Gx * dy > 0, it corresponds to a forward chaining,
    when Gy * dx - Gx * dy < 0, it corresponds to a backward chaining.

     first check that the gradient at both points to be chained agree
     in one direction, otherwise return invalid chaining. */
    const auto delta       = edgeTo - edgeFrom;
    const auto fromProject = gradFrom[ 1 ] * delta.x - gradFrom[ 0 ] * delta.y;
    const auto toProject   = gradTo[ 1 ] * delta.x - gradTo[ 0 ] * delta.y;
    if (fromProject * toProject <= 0.0) {
        return 0.0; /* incompatible gradient angles, not a valid chaining */
    }

    /* return the chaining score: positive for forward chaining,negative for backwards.
    the score is the inverse of the distance to the chaining point, to give preference to closer
    points */
    const auto dist = cv::norm(delta);
    if (fromProject >= 0.0) {
        return 1.0f / dist; /* forward chaining  */
    }
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
void chain_edge_points(cv::Mat &next, cv::Mat &prev, const cv::Mat &edge, const cv::Mat &grad) {
    const int X = edge.size().width;
    const int Y = edge.size().height;

    /* try each point to make local chains */
    for (int y = 2; y < (Y - 2); y++) { /* 2 pixel margin to include the tested neighbors */
        for (int x = 2; x < (X - 2); x++) {
            const cv::Point pos(x, y); /* edge point to be chained			*/
                                       /* must be an edge point */
            const auto &posEdge = edge.at<cv::Point2f>(pos);
            if (posEdge.x < 0.0) {
                continue;
            }

            float     forwardScore  = 0.0;      /* score of best forward chaining		*/
            float     backwardScore = 0.0;      /* score of best backward chaining		*/
            cv::Point posForward    = {-1, -1}; /* edge point of best forward chaining */
            cv::Point posBackward   = {-1, -1}; /* edge point of best backward chaining*/

            /* try all neighbors two pixels apart or less.
            looking for candidates for chaining two pixels apart, in most such cases,
            is enough to obtain good chains of edge points that	accurately describes the edge.
            */

            const auto &posGrad = grad.at<cv::Vec2s>(pos);
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    if (i == 0 && j == 0) {
                        continue;
                    }

                    cv::Point  neighbor(x + i, y + j); /* candidate edge point to be chained */
                    const auto score =
                        chain(posEdge, posGrad, neighbor, edge, grad); /* score from-to */

                    if (score > forwardScore) /* a better forward chaining found    */
                    {
                        forwardScore = score; /* set the new best forward chaining  */
                        posForward   = neighbor;
                    }
                    if (score < backwardScore) /* a better backward chaining found	  */
                    {
                        backwardScore = score; /* set the new best backward chaining */
                        posBackward   = neighbor;
                    }
                }
            }

            auto &posNext = next.at<cv::Point>(pos);
            if (posForward.x >= 0 && posNext != posForward) {
                const auto &forwardPre = prev.at<cv::Point>(posForward);
                if (forwardPre.x < 0 || chain(forwardPre, posForward, edge, grad) < forwardScore) {

                    /* remove previous from-x link if one */
                    /* only prev requires explicit reset  */
                    /* set next of from-fwd link          */
                    if (posNext.x >= 0) {
                        prev.at<cv::Point>(posNext) = {-1, -1};
                    }
                    posNext = posForward;

                    /* remove alt-fwd link if one         */
                    /* only next requires explicit reset  */
                    /* set prev of from-fwd link          */
                    if (forwardPre.x >= 0) {
                        next.at<cv::Point>(forwardPre) = {-1, -1};
                    }
                    prev.at<cv::Point>(posForward) = pos;
                }
            }

            auto &posPre = prev.at<cv::Point>(pos);
            if (posBackward.x >= 0 && posPre != posBackward) {
                const auto &backwardNext = next.at<cv::Point>(posBackward);
                if (backwardNext.x < 0 ||
                    chain(backwardNext, posBackward, edge, grad) > backwardScore) {

                    /* remove bck-alt link if one         */
                    /* only prev requires explicit reset  */
                    /* set next of bck-from link          */
                    if (backwardNext.x >= 0) {
                        prev.at<cv::Point>(backwardNext) = {-1, -1};
                    }
                    next.at<cv::Point>(posBackward) = pos;

                    /* remove previous x-from link if one */
                    /* only next requires explicit reset  */
                    /* set prev of bck-from link          */
                    if (posPre.x >= 0) {
                        next.at<cv::Point>(posPre) = {-1, -1};
                    }
                    posPre = posBackward;
                }
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
void thresholds_with_hysteresis(std::vector<std::list<cv::Point2f>> &points,
                                std::vector<std::list<cv::Vec2f>>   &dirs,
                                cv::Mat                             &next,
                                cv::Mat                             &prev,
                                const cv::Mat                       &mag,
                                const cv::Mat                       &edge,
                                const cv::Mat                       &grad,
                                float                                high) {
    const int X  = mag.size().width;
    const int Y  = mag.size().height;
    high        *= high;

    /* validate all edge points over th_h or connected to them and over th_l */
    for (int row = 0; row < Y; row++) { /* prev[i]>=0 or next[i]>=0 implies an edge point */
        for (int col = 0; col < X; col++) {
            const cv::Point pos{col, row};
            auto            mod = (float)mag.at<unsigned short>(pos);
            if (mod >= high) {
                std::list<cv::Point2f> path;
                std::list<cv::Vec2f>   dir;

                path.emplace_back(edge.at<cv::Point2f>(pos));
                const auto &posGrad = grad.at<cv::Vec2s>(pos);
                dir.emplace_back(posGrad[ 0 ] / mod, posGrad[ 1 ] / mod);

                /* follow the chain of edge points forwards */
                cv::Point lastPos = pos;
                cv::Point nextPos(-1, -1);
                std::swap(next.at<cv::Point>(lastPos), nextPos);
                while (nextPos.x >= 0) {
                    mod = (float)mag.at<unsigned short>(nextPos);
                    path.emplace_back(edge.at<cv::Point2f>(nextPos));
                    const auto &posGrad = grad.at<cv::Vec2s>(nextPos);
                    dir.emplace_back(posGrad[ 0 ] / mod, posGrad[ 1 ] / mod);

                    lastPos = nextPos;
                    nextPos = {-1, -1};
                    std::swap(next.at<cv::Point>(lastPos), nextPos);
                }

                /* follow the chain of edge points backwards */
                lastPos = pos;
                cv::Point prePos(-1, -1);
                std::swap(prev.at<cv::Point>(lastPos), prePos);
                while (prePos.x >= 0) {
                    mod = (float)mag.at<unsigned short>(prePos);
                    path.emplace_back(edge.at<cv::Point2f>(prePos));
                    const auto &posGrad = grad.at<cv::Vec2s>(prePos);
                    dir.emplace_back(posGrad[ 0 ] / mod, posGrad[ 1 ] / mod);

                    lastPos = prePos;
                    prePos  = {-1, -1};
                    std::swap(prev.at<cv::Point>(lastPos), prePos);
                }

                points.emplace_back(std::move(path));
                dirs.emplace_back(std::move(dir));
            }
        }
    }
}

void EdgePoint(const cv::Mat                       &img,
               std::vector<std::list<cv::Point2f>> &points,
               std::vector<std::list<cv::Vec2f>>   &dirs,
               float                                sigma,
               float                                low,
               float                                high) {

    cv::Mat blured;
    cv::Mat mag;
    cv::Mat grad;

    auto start = cv::getTickCount();
    cv::GaussianBlur(img, blured, cv::Size(5, 5), 0);
    gradient(blured, grad, mag);

    {
        auto end  = cv::getTickCount();
        auto cost = (end - start) / cv::getTickFrequency();
        std::cout << "preprocess cost(s):" << cost << std::endl;
    }

    // inter
    cv::Mat edge(img.size(), CV_32FC2, {-1.f, -1.f});

    cv::Mat next(img.size(), CV_32SC2, {-1.f, -1.f});
    cv::Mat prev(img.size(), CV_32SC2, {-1.f, -1.f});

    start = cv::getTickCount();
    compute_edge_points(edge, mag, grad, low);
    chain_edge_points(next, prev, edge, grad);
    thresholds_with_hysteresis(points, dirs, next, prev, mag, edge, grad, high);
    // list_chained_edge_points(points, dirs, next, prev, edge, dx2, dy2, mag);
    {
        auto end  = cv::getTickCount();
        auto cost = (end - start) / cv::getTickFrequency();
        std::cout << "cost(s):" << cost << std::endl;
    }
}