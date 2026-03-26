# 这也是R语言的函数模板，可以自定义函数。
data.outline=function(x)
  ## input ###################
  ## x = a vector ############
  ## output ##################
  ## the outline of data x ###
{  
   n  = length(x)

   m  = mean(x)
   v  = var(x)
   s  = sd(x)
   me = median(x)
   
   
   m1 = min(x)
   m2 = max(x)
   Q1 = quantile(x, 1/4)
   Q3 = quantile(x, 3/4)
   
   R  = m2-m1
   R1 = Q3-Q1
   
   cv   = s/m
   skew = sum((x-m)^3/s^3)/n
   kurt = sum((x-m)^4/s^4)/n-3
   
   return(list(size=n, Mean=m, Var=v, Std=s, Median=me, 
               Min=m1, Max=m2, Q1=Q1, Q3=Q3, R=R, R1=R1, 
               CV=cv, Skew=skew, kurtosis=kurt))
}