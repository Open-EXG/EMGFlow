# 创建目录
mkdir -p /bigdata/emgdata_private/NinaPro_DB2
cd /bigdata/emgdata_private/NinaPro_DB2

# 循环下载 S1 - S40
# -x 16: 16个线程
# -s 16: 16个连接
# -k 1M: 最小切片大小
for i in {1..40}; do
    echo "正在暴力加速下载 Subject $i ..."
    aria2c -x 16 -s 16 -k 1M -c \
           "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s${i}.zip" \
           --header="Referer: https://ninapro.hevs.ch/instructions/DB2.html" \
           -o "DB2_s${i}.zip"
done