/*
Post analysis with saved g2o files
author: Yun Chang
*/

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

double calculate_ape(const gtsam::Values& ground_truth,
                     const gtsam::Values& estimate) {
  double error = 0;
  size_t compared_poses = 0;
  gtsam::KeyVector keys = estimate.keys();
  for (gtsam::Key i : keys) {
    if (ground_truth.exists(i)) {
      compared_poses++;
      gtsam::Pose3 diff = estimate.at<gtsam::Pose3>(i).between(
          ground_truth.at<gtsam::Pose3>(i));
      gtsam::Vector log_diff = gtsam::Pose3::Logmap(diff);
      error = error + std::sqrt(log_diff.transpose() * log_diff);
    }
  }
  return error / compared_poses;
}

double calculate_odom_consistency(
    const gtsam::Values& ground_truth,
    const gtsam::Values& estimate,
    const gtsam::NonlinearFactor::shared_ptr& new_factor) {
  gtsam::Key front = new_factor->front();
  gtsam::Key back = new_factor->back();

  gtsam::BetweenFactor<gtsam::Pose3> odom_factor =
      *boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(
          new_factor);

  gtsam::Pose3 between = odom_factor.measured();
  gtsam::Pose3 between_gt = ground_truth.at<gtsam::Pose3>(front).between(
      ground_truth.at<gtsam::Pose3>(back));

  gtsam::Vector log = gtsam::Pose3::Logmap(between.between(between_gt));
  double error = std::sqrt(log.transpose() * log);

  return error / (front - back);
}

void separate_odometry_from_lc(const gtsam::NonlinearFactorGraph& input,
                               gtsam::NonlinearFactorGraph* odom_nfg,
                               gtsam::NonlinearFactorGraph* lc_nfg) {
  for (size_t i = 0; i < input.size(); i++) {
    if (NULL == input[i]) continue;
    // check if factor is a between factor
    if (boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(
            input[i])) {
      if (input[i]->front() + 1 == input[i]->back()) {
        odom_nfg->add(input[i]);
      } else {
        lc_nfg->add(input[i]);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  // usage: ./PostAnalysis gt_g2o results_folder

  // Load ground truth
  gtsam::GraphAndValues graphNValues_gt = gtsam::load3D(argv[1]);
  gtsam::Values values_gt = *graphNValues_gt.second;

  // Load result
  std::string result_path = std::string(argv[2]) + "/result.g2o";
  gtsam::GraphAndValues graphNValues = gtsam::load3D(result_path);
  gtsam::NonlinearFactorGraph nfg = *graphNValues.first;
  gtsam::Values values = *graphNValues.second;

  // Load the odom inconsistent factors
  std::string odom_inc_path = std::string(argv[2]) + "/odom_inconsistent.g2o";
  gtsam::GraphAndValues graphNValues_oi = gtsam::load3D(odom_inc_path);
  gtsam::NonlinearFactorGraph odom_rejected_factors = *graphNValues_oi.first;

  // Load the pairwise inconsistent factors
  std::string pw_inc_path = std::string(argv[2]) + "/pairwise_inconsistent.g2o";
  gtsam::GraphAndValues graphNValues_pi = gtsam::load3D(pw_inc_path);
  gtsam::NonlinearFactorGraph pw_rejected_factors = *graphNValues_pi.first;

  gtsam::NonlinearFactorGraph odom_nfg;
  gtsam::NonlinearFactorGraph inlier_factors;

  separate_odometry_from_lc(nfg, &odom_nfg, &inlier_factors);

  std::string output_file = std::string(argv[2]) + "/post_analysis.txt";
  std::ofstream outfile;
  outfile.open(output_file, std::ios::out);

  outfile
      << "from to ape odom_consistency label(0=inliers 1=pw_incon 2=odom_rej)"
      << std::endl;

  // First loop through inliers and calculate ape
  for (size_t i = 0; i < inlier_factors.size(); i++) {
    gtsam::NonlinearFactorGraph test_nfg;
    test_nfg.add(odom_nfg);
    test_nfg.add(inlier_factors[i]);

    // optimize
    gtsam::LevenbergMarquardtParams params;
    params.diagonalDamping = true;
    gtsam::Values test_values =
        gtsam::LevenbergMarquardtOptimizer(test_nfg, values, params).optimize();
    double ape = calculate_ape(values_gt, test_values);

    double odom_consistency =
        calculate_odom_consistency(values_gt, test_values, inlier_factors[i]);

    gtsam::Key front = inlier_factors[i]->front();
    gtsam::Key back = inlier_factors[i]->back();

    outfile << front << " " << back << " " << ape << " " << odom_consistency
            << " " << 0 << std::endl;
  }

  // Then loop through pairwise inconsistent and calculate ape
  for (size_t i = 0; i < pw_rejected_factors.size(); i++) {
    gtsam::NonlinearFactorGraph test_nfg;
    test_nfg.add(odom_nfg);
    test_nfg.add(pw_rejected_factors[i]);

    // optimize
    gtsam::LevenbergMarquardtParams params;
    params.diagonalDamping = true;
    gtsam::Values test_values =
        gtsam::LevenbergMarquardtOptimizer(test_nfg, values, params).optimize();
    double ape = calculate_ape(values_gt, test_values);

    double odom_consistency = calculate_odom_consistency(
        values_gt, test_values, pw_rejected_factors[i]);

    gtsam::Key front = pw_rejected_factors[i]->front();
    gtsam::Key back = pw_rejected_factors[i]->back();

    outfile << front << " " << back << " " << ape << " " << odom_consistency
            << " " << 1 << std::endl;
  }

  // Last loop through odom inconsistent and calculate ape
  for (size_t i = 0; i < odom_rejected_factors.size(); i++) {
    gtsam::NonlinearFactorGraph test_nfg;
    test_nfg.add(odom_nfg);
    test_nfg.add(odom_rejected_factors[i]);

    // optimize
    gtsam::LevenbergMarquardtParams params;
    params.diagonalDamping = true;
    gtsam::Values test_values =
        gtsam::LevenbergMarquardtOptimizer(test_nfg, values, params).optimize();
    double ape = calculate_ape(values_gt, test_values);

    double odom_consistency = calculate_odom_consistency(
        values_gt, test_values, odom_rejected_factors[i]);

    gtsam::Key front = odom_rejected_factors[i]->front();
    gtsam::Key back = odom_rejected_factors[i]->back();

    outfile << front << " " << back << " " << ape << " " << odom_consistency
            << " " << 2 << std::endl;
  }

  outfile.close();
}
