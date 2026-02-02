function data = load_mat(pth)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
data1 = load(pth);
data = data1.spec_avg;
end