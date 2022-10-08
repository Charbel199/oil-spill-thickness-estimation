%% This code is to generate reflectivities based on the provided frequencies and thicknesses
%% The code is also able to add normal gaussian noise to the generated reflectivities
% -----------------------------------------------------------------------

clear all

%% Parameters
% -----------------------------------------------------------------------

% Output size
number_of_observations = 1;
number_of_samples = 10000;
output_file_type = "txt";

% Environment parameters
temperature_w = 20; % water temperature in degrees
salinity_w = 30; % salinity in parts per thousand psu
c = 3e8; % speed of light
g = 9.81; % gravity constant
wind_speeds = [2:2:16]; % Wind speed(10 m above sea) from 2 to 16 m/s 
rms_heights= [7e-4 7e-4 8e-4 1e-3 1.5e-3 1.6e-3 1.7e-3 1.8e-3]; % RMS Surface height based on https://ieeexplore.ieee.org/abstract/document/1356073


variance_noise= 0.02; % Gaussian noise variance

% Frequencies
%f= [4.3855 6.9759 9.0681]*1e9; % frequency range of EM in GHz  4.151,7.743,7.939 or 11
f= [3:1:14]*1e9;
%f= [4 4.5 5 6 7 7.5 8 9 10 10.5 11 12]*1e9;
lambda= c./f; % EM wavelength

% Surface roughness
wind_speed = 8;
s = rms_heights(wind_speeds ==  wind_speed);
rough_surface = true;
ks=zeros(1, length(f));
if rough_surface
    ks = (2*pi./lambda).*s;
    loss = exp(-4.*(ks*cosd(0)).^2)
end


% Thicknesses range
thickness= [0:1:10]; % in mm

% Permittivity range of medium 2, oil
permittivity_range = [3];

% Dielectric permmittivities of medium 1 and 2
[epsr_w, epsi_w] = module4_2(temperature_w,f/1e9,salinity_w);
eps1 = 1; % dielectric permittivity of medium 1, air
eps3 = epsr_w-epsi_w*1i; % dielectric permittivity of medium 3, saline water

% Base file name
file_name = strcat("thickness-",num2str(length(f),2),"freqs-variance",num2str(variance_noise, 2),"-");

% -----------------------------------------------------------------------
%% 



% Reflectivity generation
% -----------------------------------------------------------------------
for eps2= permittivity_range
    n1= sqrt(eps1);
    n2= sqrt(eps2);
    n3= sqrt(eps3);

    r = zeros(length(lambda),length(thickness));
    R = zeros(length(lambda),length(thickness)); % Number of frequencies x Number of thicknesses


    for p=1:length(lambda) % Iterate through each wavelength
        % Reflection coefficient for single interfaces
        r1 = (n1-n2)/(n1+n2); % air-oil interface
        r2(p)= (n2-n3(p))/(n2+n3(p)); % oil-water interface
        r_free(p) = (r1+r2(p))./(1+r1*r2(p)); % refl coef
        R_free(p)= (abs(r_free(p)))^2*loss(p);

        % Variation in Oil Thickness
        delta2 = 2*pi/lambda(p)*n2*0.001.*(thickness); % for 1 frequency value
        r(p,:) = (r1+r2(p).*exp(-1j*2.*delta2))./(1+r1*r2(p).*exp(-1j*2.*delta2)); % refl coef
        R(p,:)= (abs(r(p,:))).^2*loss(p); % Reflectivity
    end
    
    % For each thickness
    for thickness_index=1:length(thickness)
        % Generate reflectivities with noise 
        reflectivities_with_noise = noise(R(:,thickness_index), variance_noise, number_of_observations, number_of_samples);
        % Export reflectivities with noise
        thickness_file_name = strcat(file_name,num2str(thickness(thickness_index)));
        export_to_file(squeeze(reflectivities_with_noise), thickness_file_name, size(R(:,thickness_index),1), output_file_type);
    end
end
% -----------------------------------------------------------------------