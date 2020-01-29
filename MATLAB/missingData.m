% Processing Data in MATLAB
% Authors: Andrea Murphy
% Date: January 28, 2020
% DESC: Make and organize missing data

% Both functions create the same missing values
x = [NaN 1 2 3 4];
xDouble = [missing 1 2 3 4];

% Missing date and time data
xDatetime = [missing datetime(2014,1:4,1)];

% Missing string data
xstring = [NaN "a" "b" "c" "d"];

% Sort missing data
xSort = sort(xStandard,'MissingPlacement','last');

% Find, replace, and ignore missing data
nanData = [1:9 NaN];
plot(1:10,nanData)

