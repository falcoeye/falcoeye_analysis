source ~/.bash_profile
pods | grep '\-\-' | awk '{print $1}' | xargs kubectl  delete pod
