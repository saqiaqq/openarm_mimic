#pragma once

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

class RobotKDL {
public:
    RobotKDL(const std::string& urdf_content, const std::string& base_link, const std::string& tip_link);
    ~RobotKDL() = default;

    bool init();
    
    // Inverse Kinematics: Cartesian Pose -> Joint Angles
    // q_init: initial guess (usually current joint positions)
    // target_pose: [x, y, z, qx, qy, qz, qw]
    // q_out: result joint angles
    bool solveIK(const std::vector<double>& q_init, 
                 const std::vector<double>& target_pose, 
                 std::vector<double>& q_out);

    // Forward Kinematics: Joint Angles -> Cartesian Pose
    bool solveFK(const std::vector<double>& q_in,
                 std::vector<double>& pose_out);

    int getNumJoints() const { return num_joints_; }

private:
    std::string urdf_content_;
    std::string base_link_;
    std::string tip_link_;
    int num_joints_;

    KDL::Tree tree_;
    KDL::Chain chain_;
    
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_;
};
