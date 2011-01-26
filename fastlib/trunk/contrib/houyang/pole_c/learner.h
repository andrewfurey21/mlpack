#ifndef LEARNER_H
#define LEARNER_H

#include "loss_functions.h"
#include "weak_learners.h"
#include "sparsela.h"

// using namespace boost::posix_time;

struct learner {
  SVEC** w_vec_pool; // a pool that contains weight vectors for each thread
  SVEC** w_n_vec_pool; // a pool that contains weight_negative vectors for each thread; used for EG only.
  double* bias_pool; // a pool for bias term
  double* bias_n_pool; // a pool for bias_negative term; used for EG only.
  SVEC** msg_pool; // a pool of messages. each thread put its message in and read other's from
  double* t_pool; // time t for SGD
  double* scale_pool; // scales for SGD
  size_t* num_used_exp; // count the number of examples used by a thread
  int reg; // Which regularization term to use; 1:L1, 2:squared L2(default), -1: no regularization
  double reg_factor; // regularization weight ('lambda' in avg_loss + lambda * regularization)
  double C; // cost factor C (regularization + C * sum_loss)
  size_t num_threads;
  size_t num_epoches;
  string type; // classification, or regression, or others
  string opt_method;
  string loss_name;
  double* total_loss_pool; // a pool of total loss for each thread;
  size_t* total_misp_pool; // a pool of total number of mispredictions for each thread;
  LossFunctions *loss_func;

  size_t num_experts; // number of experts for ensemble methods (WM)
  size_t **expert_misp; // number of mispredictions for each expert, over each agent
  WeakLearners **weak_learners; // basis learners
  string wl_name;
  double alpha; // the multiplication factor in WM

  size_t num_log; // how many log points
  size_t t_int; // intervals for log points
  size_t *t_ct; // counters for round t
  size_t *lp_ct; // counters for log points
  size_t **log_err; // for logging error of each thread
  double **log_loss; // for logging loss of each thread
  /*
  ptime t_start, *t_end; // for logging cpu time
  time_duration *t_duration; // for logging cpu time
  long *t_ms; // t_duration in ms
  */
};

double LinearPredict(SVEC *wvec, EXAMPLE *ex) {
  return SparseDot(wvec, ex);
}

T_LBL LinearPredictLabel(SVEC *wvec, EXAMPLE *ex) {
  double sum = SparseDot(wvec, ex);
  if (sum > 0.0)
    return (T_LBL)1;
  else
    return (T_LBL)-1;
}

double LinearPredictBias(SVEC *wvec, EXAMPLE *ex, double bias) {
  //print_svec(wvec);
  //print_ex(ex);
  //cout << bias << endl;
  return SparseDot(wvec, ex) + bias;
}

T_LBL LinearPredictBiasLabel(SVEC *wvec, EXAMPLE *ex, double bias) {
  double sum = SparseDot(wvec, ex) + bias;
  /*print_svec(wvec);
  print_ex(ex);
  cout << sum<<endl;
  */
  if (sum > 0.0)
    return (T_LBL)1;
  else
    return (T_LBL)-1;
}

void FinishLearner(learner &l, size_t ts) {
  if(l.w_vec_pool) {
    for (size_t t=0; t<ts; t++) {
      //print_svec(l.w_vec_pool[t]);
      DestroySvec(l.w_vec_pool[t]);
    }
    free(l.w_vec_pool);
  }
  if (l.opt_method == "oeg") {
    if(l.w_n_vec_pool) {
      for (size_t t=0; t<ts; t++) {
	//print_svec(l.w_vec_pool[t]);
	DestroySvec(l.w_n_vec_pool[t]);
      }
      free(l.w_n_vec_pool);
    }
  }
  if(l.msg_pool) {
    for (size_t t=0; t<ts; t++) {
      DestroySvec(l.msg_pool[t]);
    }
    free(l.msg_pool);
  }
  free(l.t_pool);
  free(l.scale_pool);
  free(l.bias_pool);
  free(l.bias_n_pool);
  free(l.num_used_exp);
  free(l.total_loss_pool);
  free(l.total_misp_pool);
}

void TickWait(size_t ticks) {
  clock_t t = ticks + clock();
  while (t > clock());
}

#endif
