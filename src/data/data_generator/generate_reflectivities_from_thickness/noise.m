function [reflectivities_with_noise]= noise(R, variance_noise, number_of_observations, number_of_samples) 
% R: Theoretical reflectivities for a specific thickness
% variance_noise: Variance
% number_of_observations: Number of observations
% number_of_samples: Number of samples with same thickness
% --------------------------------------------------------------
%                   MONTE CARLO SIMULATION
%  --------------------------------------------------------------



number_of_frequencies = size(R,1);
reflectivities = zeros(number_of_frequencies, number_of_observations, number_of_samples);
sigma = sqrt(variance_noise);

for frequency=1:number_of_frequencies
    for n=1:number_of_observations
        reflectivities(frequency,n,:) = R(frequency)+sigma.*randn(1,number_of_samples);
    end
end

% Compute the mean of the all observations
reflectivities_with_noise = squeeze(mean(reflectivities, 2));






