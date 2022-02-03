function [reflectivities_with_noise]= noise(R, thickness, variance_noise) 
% R: Theoretical reflectivities for a specific thickness
% thickness: Corresponding thickness
% variance_noise: Variance
% --------------------------------------------------------------
%                   MONTE CARLO SIMULATION
%  --------------------------------------------------------------

N=1; % number of observations
Grid_len = 10000; % Number of samples with same thickness

number_of_frequencies = size(R,1);
reflectivities_with_noise = zeros(number_of_frequencies, N,Grid_len);
sigma = sqrt(variance_noise);

for frequency=1:number_of_frequencies
    for n=1:N
        reflectivities_with_noise(frequency,n,:) = R(frequency)+sigma.*randn(1,Grid_len);
    end
end
  





