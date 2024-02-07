#include <gpu_bbs3d/bbs3d.cuh>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

std::vector<Eigen::Vector3f> createTransSearchRange(const Eigen::Vector3f& center_pose,
                                                                         float width)
{
  std::vector<Eigen::Vector3f> trans_search_range(2);
  trans_search_range[0] << center_pose.x() + width, center_pose.y() + width, center_pose.z() + width;
  trans_search_range[1] << center_pose.x() - width, center_pose.y() - width, center_pose.z() - width;
  return trans_search_range;
};

Eigen::Vector3f rpyRange2Eigen(const std::vector<float>& vec)
{
  Eigen::Vector3f e_vec;
  for (int i = 0; i < 3; ++i)
  {
    if (vec[i] == 6.28)
    {
      e_vec(i) = 2 * M_PI;
    }
    else
    {
      e_vec(i) = vec[i];
    }
  }
  return e_vec;
};

void pcl2Eigen(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud, std::vector<Eigen::Vector3f>& points)
{
  points.resize(pcl_cloud->points.size());
  std::transform(pcl_cloud->begin(), pcl_cloud->end(), points.begin(), [](const pcl::PointXYZ& p) { return Eigen::Vector3f(p.x, p.y, p.z); });
}


int main(int argc, char** argv)
{
  std::string target_cloud_path(argv[1]);
  std::string source_cloud_path(argv[2]);

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(target_cloud_path, *target_cloud) == -1)
  {
    std::cerr << "Failed to load target cloud " << target_cloud_path << std::endl;
    exit(1);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile(source_cloud_path, *source_cloud) == -1)
  {
    std::cerr << "Failed to load source cloud " << source_cloud_path << std::endl;
    exit(1);
  }

  std::unique_ptr<gpu::BBS3D> bbs3d_ptr = std::make_unique<gpu::BBS3D>();
  std::vector<Eigen::Vector3f> target_points;
  pcl2Eigen(target_cloud, target_points);

  bbs3d_ptr->set_tar_points(target_points, 0.5, 6);
  bbs3d_ptr->set_angular_search_range(rpyRange2Eigen(std::vector<float>({-0.02, -0.02, 0.0})), rpyRange2Eigen(std::vector<float>({0.02, 0.02, 6.28})));

  Eigen::Vector3f input_pose(10.74, -0.995, 0.0);
  auto trans_search_range = createTransSearchRange(input_pose, 20.0);
  bbs3d_ptr->set_trans_search_range(trans_search_range);

  std::vector<Eigen::Vector3f> source_points;
  pcl2Eigen(source_cloud, source_points);

  bbs3d_ptr->set_src_points(source_points);
  bbs3d_ptr->set_score_threshold_percentage(0.5);

  bbs3d_ptr->localize();

  std::cout << "Score: " << bbs3d_ptr->get_best_score() << std::endl;

  return 0;
}
