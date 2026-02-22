#ifndef OPENARM_MIMIC_ROBOT_KDL_HPP
#define OPENARM_MIMIC_ROBOT_KDL_HPP

/**
 * @file robot_kdl.hpp
 * @brief 机器人运动学库封装头文件
 * 
 * 此文件定义了 RobotKDL 类，用于基于 KDL (Kinematics and Dynamics Library)
 * 实现机械臂的正向运动学 (FK) 和逆向运动学 (IK) 求解。
 */

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl_parser/kdl_parser.hpp>

/**
 * @class RobotKDL
 * @brief 封装 KDL 库以提供便捷的运动学求解接口
 */
class RobotKDL {
public:
    /**
     * @brief 构造函数
     * @param urdf_content 机器人的 URDF 模型内容字符串
     * @param base_link 运动学链的基座连杆名称
     * @param tip_link 运动学链的末端连杆名称
     */
    RobotKDL(const std::string& urdf_content, const std::string& base_link, const std::string& tip_link);
    
    /**
     * @brief 析构函数
     */
    ~RobotKDL() = default;

    /**
     * @brief 初始化 KDL 求解器
     * 
     * 解析 URDF，构建 KDL Tree 和 Chain，并初始化 FK/IK 求解器。
     * @return true 初始化成功
     * @return false 初始化失败
     */
    bool init();

    /**
     * @brief 求解逆向运动学 (Inverse Kinematics)
     * 
     * 根据当前关节角猜测值和目标末端位姿，计算目标关节角。
     * 
     * @param q_init 当前关节角（作为求解的初值猜测），单位：弧度
     * @param target_pose 目标末端位姿 [x, y, z, qx, qy, qz, qw]
     * @param q_out [输出] 计算得到的关节角，单位：弧度
     * @return true 求解成功
     * @return false 求解失败
     */
    bool solveIK(const std::vector<double>& q_init, 
                 const std::vector<double>& target_pose, 
                 std::vector<double>& q_out);

    /**
     * @brief 求解正向运动学 (Forward Kinematics)
     * 
     * 根据给定的关节角，计算末端执行器的位姿。
     * 
     * @param q_in 输入关节角，单位：弧度
     * @param pose_out [输出] 计算得到的末端位姿 [x, y, z, qx, qy, qz, qw]
     * @return true 求解成功
     * @return false 求解失败
     */
    bool solveFK(const std::vector<double>& q_in, std::vector<double>& pose_out);

    /**
     * @brief 获取关节数量
     * @return int 关节数量
     */
    int getNumJoints() const { return num_joints_; }

private:
    std::string urdf_content_;  ///< URDF 内容字符串
    std::string base_link_;     ///< 基座连杆名
    std::string tip_link_;      ///< 末端连杆名
    int num_joints_;            ///< 关节自由度数量

    KDL::Tree tree_;            ///< KDL 树结构
    KDL::Chain chain_;          ///< KDL 运动学链
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_; ///< 正运动学求解器
    std::unique_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_;       ///< 逆运动学求解器 (LMA算法)
};

#endif // OPENARM_MIMIC_ROBOT_KDL_HPP
