#include <map>
#include <Rcpp.h>
using namespace Rcpp;

class Stat {
public:
  double sum;
  double ssq;
  int cnt;

  Stat(): sum(0), cnt(0) {}
};

//[[Rcpp::export]]
List group_avg(NumericVector x, NumericVector y, double interval) {
  std::map<int, Stat> m;
  assert(x.size() == y.size());
  int n = x.size();
  for(int i=0; i<n; i++) {
    int idx = x[i]/interval;
    std::map<int,Stat>::iterator iter = m.find(idx);
    if(iter != m.end()) {
      iter->second.sum += y[i];
      iter->second.cnt += 1;
    }
    else {
      m[idx].sum = y[i];
      m[idx].ssq = y[i] * y[i];
      m[idx].cnt = 1;
    }
  }
  std::vector<double> rx, ry, rsd;
  for(std::map<int,Stat>::iterator iter=m.begin(); iter!=m.end(); ++iter) {
    rx.push_back(iter->first * interval);
    ry.push_back(iter->second.sum / iter->second.cnt);
    rsd.push_back((iter->second.ssq - iter->second.sum * iter->second.sum) / iter->second.cnt);
  }
  return List::create(Named("x") = rx, Named("y") = ry, Named("sd") = rsd);
}

