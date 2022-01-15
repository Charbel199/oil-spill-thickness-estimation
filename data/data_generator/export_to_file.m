function []= export_to_file(reflectivities, file_name, number_of_frequencies, thickness) 
% reflectivities_with_noise: Reflectivities to add to the file
% file_name: Beginning of the file name
% number_of_frequencies: Total number of frequencies
% thickness: Current thickness

% Save data into files
file_name = strcat(file_name,num2str(thickness),'.txt');
file=fopen(file_name,'w');
for i=1: length(reflectivities)
    row = '';
    for frequency=1:number_of_frequencies
        row = row + " " + num2str(reflectivities(frequency,i));
    end
    row = row + "\n";
    fprintf(file,row);
end

fclose(file);



end