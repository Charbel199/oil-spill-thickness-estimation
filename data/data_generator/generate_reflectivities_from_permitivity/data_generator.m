%% This code is to generate reflectivities based on the provided frequencies and thicknesses
%% The code is also able to add normal gaussian noise to the generated reflectivities
% -----------------------------------------------------------------------

clear all

% Parameters
%f= [4.3855 6.9759 9.0681]*1e9; % frequency range of EM in GHz  4.151,7.743,7.939 or 11
%f= [1 2 3 4 5 6 7 8 9 10 11 12]*1e9;
f= [4 7 10 12]*1e9;
ks=[0 0 0 0]; 
loss = exp(-4.*(ks*cosd(0)).^2);
%s=0.3e-2;
%ks= (2*pi./lambda).*s
%thickness= [0:1:10]; % in mm
thickness = 9;
variance_noise= 0.02;
file_name = "9mm-50observations-4freqs-variance0.02-";
N=50; % number of observations

c = 3e8; % speed of light
lambda= c./f; % EM wavelength

temperature_w = 20; % water temperature in degrees
salinity_w = 30; % salinity in parts per thousand psu
[epsr_w, epsi_w] = module4_2(temperature_w,f/1e9,salinity_w);
eps1 = 1; % dielectric permittivity of medium 1, air
eps3 = epsr_w-epsi_w*1i; % dielectric permittivity of medium 3, saline water
eps2= [1.9:0.1:3.3]; % dielectric permittivity of medium 2, oil
n1 = sqrt(eps1);
n2= sqrt(eps2);
n3= sqrt(eps3);


r = zeros(length(lambda),length(eps2));
R = zeros(length(lambda),length(eps2)); % Number of frequencies x Number of permitivities


for p=1:length(lambda) % variation in frequency
    % reflection coefficient for single interfaces
    r1 = (n1-n2)/(n1+n2); % air-oil interface
    r2(p)= (n2-n3(p))/(n2+n3(p)); % oil-water interface
    r_free(p) = (r1+r2(p))./(1+r1*r2(p)); % refl coef
    %R_free(p)= (abs(r_free(p)))^2*loss(p);

    % Variation in Oil Thickness
    delta2 = 2*pi/lambda(p)*n2*0.001.*(thickness); % for 1 frequency value
    r(p,:) = (r1+r2(p).*exp(-1j*2.*delta2))./(1+r1*r2(p).*exp(-1j*2.*delta2)); % refl coef
    R(p,:)= (abs(r(p,:))).^2*loss(p); % Reflectivity
end

for s=1:length(eps2)
    reflectivities_with_noise = noise(R(:,s),eps2(s), variance_noise, N);
    export_to_file(reflectivities_with_noise, file_name, size(R(:,s),1), eps2(s));
end
