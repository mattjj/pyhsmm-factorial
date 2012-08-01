using namespace Eigen;

// inputs

double enoise_variance = noise_variance;
Map<MatrixXd> evarseq(varseq,%(K)d,%(T)d);
Map<MatrixXd> emeanseq(meanseq,%(K)d,%(T)d);
// for now, calculate these in numpy, too
Map<MatrixXd> epost_meanseq(post_meanseq,%(K)d,%(T)d);
Map<MatrixXd> eG(G,%(K)d,%(T)d);

// outputs

Map<MatrixXd> econtributions(contributions,%(K)d,%(T)d);

// local vars

MatrixXd updated_chol(%(K)d,%(K)d);
MatrixXd X(%(K)d,%(K)d);
VectorXd sumsq(%(K)d);
VectorXd ev(%(K)d), el(%(K)d);

for (int t=0; t < %(T)d; t++) {

    sumsq.setZero();
    X.setZero();

    el = evarseq.col(t).cwiseSqrt();
    ev = evarseq.col(t) / (sqrt(evarseq.col(t).sum() + enoise_variance)); // TODO make noise_variance work passed in

    // compute update into X
    for (int j = 0; j < %(K)d; j++) {
        for (int i = j; i < %(K)d; i++) {
            if (i == j) {
                X(i,i) = el(i) - sqrt(el(i)*el(i) - sumsq(i) - ev(i)*ev(i));
            } else {
                X(i,j) = (-1.*ev(i)*ev(j) - X.row(i).head(j).dot(X.row(j).head(j))) / (X(j,j) - el(j));
                sumsq(i) += X(i,j) * X(i,j);
            }
        }
    }

    // write into updated_chol
    updated_chol = el.asDiagonal();
    updated_chol -= X;

    // generate a sample
    econtributions.col(t) = updated_chol * eG.col(t) + epost_meanseq.col(t);
}
