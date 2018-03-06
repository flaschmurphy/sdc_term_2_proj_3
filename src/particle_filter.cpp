/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

   num_particles = 10;

   std::normal_distribution<double> N_x(x, std[0]);
   std::normal_distribution<double> N_y(y, std[1]);
   std::normal_distribution<double> N_theta(theta, std[2]);

   for (int i=0; i<num_particles; ++i) 
   {
       Particle particle;
       particle.id = i;
       particle.x = N_x(gen);
       particle.y = N_y(gen);
       particle.theta = N_theta(gen);
       particle.weight = 1.0;
       
       particles.push_back(particle);
   }

   is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    for (int i=0; i<num_particles; ++i)
    {
        double old_x = particles[i].x;
        double old_y = particles[i].y;
        double old_theta = particles[i].theta;

        double new_x;
        double new_y;
        double new_theta;

        if (abs(yaw_rate) > 1e-5)
        {
            new_x = old_x + (velocity/yaw_rate) * (sin(old_theta + yaw_rate * delta_t) - sin(old_theta));
            new_y = old_y + (velocity/yaw_rate) * (cos(old_theta) - cos(old_theta + yaw_rate * delta_t));
            new_theta = old_theta + yaw_rate * delta_t;
        }
        else
        {
            new_x = old_x + velocity * delta_t * cos(old_theta);
            new_y = old_y + velocity * delta_t * sin(old_theta);
            new_theta = old_theta;
        }

        normal_distribution<double> N_x(new_x, std_pos[0]);
        normal_distribution<double> N_y(new_y, std_pos[1]);
        normal_distribution<double> N_theta(new_theta, std_pos[2]);

        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto& observation : observations)
    {
        double min_dist = numeric_limits<double>::max();
        for (const auto& predicted_observation : predicted)
        {
            double distance = dist(observation.x, observation.y, predicted_observation.x, predicted_observation.y);
            if (distance < min_dist)
            {
                observation.id = predicted_observation.id;
                min_dist = distance;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // Reset weights for all particles
    for (auto& particle : particles)
        particle.weight = 1.0;

    for (int p=0; p<num_particles; ++p) 
    {
        // Assign vars for better readability
        auto particle_x = particles[p].x;
        auto particle_y = particles[p].y;
        auto particle_theta = particles[p].theta;

        // Remove all landmarks that are beyond the sensor range
        std::vector<LandmarkObs> predicted_landmarks;
        for (const auto& map_landmark : map_landmarks.landmark_list)
        {
            int    landmark_id = map_landmark.id_i;
            double landmark_x  = (double) map_landmark.x_f;
            double landmark_y  = (double) map_landmark.y_f;

            double distance = dist(particle_x, particle_y, landmark_x, landmark_y);

            if (distance < sensor_range) {
                LandmarkObs landmark_prediction;
                landmark_prediction.id = landmark_id;
                landmark_prediction.x = landmark_x;
                landmark_prediction.y = landmark_y;
                predicted_landmarks.push_back(landmark_prediction);
            }
        }

        // Convert predicted landmarks from car coordinates to map coordinates
        std::vector<LandmarkObs> landmark_observations_map_coords;

        for (auto& observed_landmark : observations)
        {
            LandmarkObs observed_landmark_map_coords;
            observed_landmark_map_coords.x = particle_x + cos(particle_theta)*observed_landmark.x
                                             - sin(particle_theta)*observed_landmark.y;
            observed_landmark_map_coords.y = particle_y + sin(particle_theta)*observed_landmark.x
                                             + cos(particle_theta)*observed_landmark.y;
            landmark_observations_map_coords.push_back(observed_landmark_map_coords);
        }

        // Landmark Associations
        dataAssociation(predicted_landmarks, landmark_observations_map_coords);

        // Initial default weight
        double new_weight = 1.0;

        // Declare new standard deviations in x and y
        double mu_x, mu_y;

        // Calculate new weight for this particle
        for (const auto& observation : landmark_observations_map_coords)
        {
            for (const auto& landmark: predicted_landmarks)
            {
                if (observation.id == landmark.id) {
                    mu_x = landmark.x;
                    mu_y = landmark.y;
                    break;
                }
            }

            double normalizer = 2 * M_PI * std_landmark[0] * std_landmark[1];

            double exponent = pow(observation.x - mu_x, 2) / (2 * std_landmark[0] * std_landmark[0]) +
                    pow(observation.y - mu_y, 2) / (2 * std_landmark[1] * std_landmark[1]);

            double multivariate = (1/normalizer) * exp(-1 * exponent);

            new_weight *= multivariate;
        }
        particles[p].weight = new_weight;
    }

    // Normalize weights
    double normalizer = 0.0;

    for (const auto& particle : particles)
        normalizer += particle.weight;

    for (auto& particle : particles)
        particle.weight /= normalizer;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<double> particle_weights;
    for (const auto& particle : particles)
        particle_weights.push_back(particle.weight);

    discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

    vector<Particle> resampled_particles;
    for (size_t i = 0; i < num_particles; ++i) {
        int k = weighted_distribution(gen);
        resampled_particles.push_back(particles[k]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
