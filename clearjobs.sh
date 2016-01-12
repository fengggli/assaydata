
ps aux|grep Drone|awk '{print $2}'|xargs -i kill -9 {}
ps aux|grep svm|awk '{print $2}'|xargs -i kill -9 {}
