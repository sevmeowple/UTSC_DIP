/// 统一文件路径处理

use std::path::{Path, PathBuf};

// 多dir拼接
pub fn join_paths(paths: &[&str]) -> PathBuf {
    let mut path = PathBuf::new();
    for p in paths {
        path.push(p);
    }
    path
}
