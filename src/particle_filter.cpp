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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 200;

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);


    weights.resize(num_particles);
    particles.resize(num_particles);

    for (int index = 0; index < num_particles; index++) {
        Particle& particle = particles[index];

        particle.id     = index;
        particle.weight = 1;
        particle.x      = dist_x(gen);
        particle.y      = dist_y(gen);
        particle.theta  = dist_theta(gen);

        weights[index]  = 1.0;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    for (auto& particle :  particles) {
        double x_mean      = particle.x;
        double y_mean      = particle.y;
        double theta_mean  = particle.theta;

        if (fabs(yaw_rate) < 0.0001) {
            x_mean    += velocity * delta_t * cos(particle.theta);
            y_mean    += velocity * delta_t * sin(particle.theta);
            //particle.theta += yaw_rate * delta_t;
        } else {
            x_mean     += (velocity/yaw_rate) * ( sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta) );
            y_mean     += (velocity/yaw_rate) * ( cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t) );
            theta_mean += yaw_rate * delta_t;    
        }

        normal_distribution<double> dist_x(x_mean, std_pos[0]);
        normal_distribution<double> dist_y(y_mean, std_pos[1]);
        normal_distribution<double> dist_theta(theta_mean, std_pos[2]);

        particle.x     = dist_x(gen);
        particle.y     = dist_y(gen);
        particle.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    for (auto& observation : observations) {
        double min_dist = numeric_limits<double>::max(); 
        for (auto& particle : predicted) {
            // Compute the distance 
            double dst = dist(observation.x, observation.y, particle.x, particle.y);

            //std::cout << particle.id << " - observation : (" << observation.x << ", " << observation.y << ") / particle("
            //          << particle.x << ", " << particle.y  << ")  : (dst : " << dst << ", min : " << min_dist << ")" << endl;

            if ((dst < min_dist)) {
                min_dist        = dst;
                observation.id  = particle.id;
            }
        }
        //std::cout << "min ID : " << observation.id  << endl;
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
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    double stdx       = std_landmark[0];
    double stdy       = std_landmark[1];
    double variance_x = pow(stdx,2);
    double variance_y = pow(stdy,2);
    double alpha      = ( 1/(2*M_PI*stdx*stdy));
    auto& landmarks   = map_landmarks.landmark_list;

    for (auto& particle : particles) {
        std::vector<LandmarkObs> predicted;
        std::vector<LandmarkObs> transformed_obs;
       
        // Ignore landmark outside sensor range
        int index = 0;
        for (auto& landmark : map_landmarks.landmark_list)
            if (fabs(landmark.x_f - particle.x) <= sensor_range && abs(landmark.y_f - particle.y) <= sensor_range)
                predicted.emplace_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});

        // transform observations coordinates
        transform(begin(observations),end(observations), back_inserter(transformed_obs), [=](const LandmarkObs& observation) -> LandmarkObs{
            double x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
            double y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;

            return LandmarkObs{observation.id, x, y};
        });

        // Associate observations
        dataAssociation(predicted, transformed_obs);

        // re-compute weight.
        particle.weight = accumulate(begin(transformed_obs), 
                                     end(transformed_obs), 
                                     1.0, [=](double total, const LandmarkObs& observation){
            auto& landmark   = landmarks[observation.id-1];
            double mux       = landmark.x_f;
            double muy       = landmark.y_f;
            double x         = observation.x;
            double y         = observation.y;
            double weight    = (alpha * exp( -( pow(x - mux,2)/(2*variance_x)  + (pow(y - muy,2)/(2*variance_y)) ) ));

            //std::cout << " Obs ID : " << observation.id  << ", Weight : "  << weight << endl;

            
            return  total * weight;
        });
        weights[particle.id] = particle.weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle>           new_particles;
    discrete_distribution<int> dist(begin(weights), end(weights));
    default_random_engine      gen;

    for(int id = 0; id < num_particles; ++id) {
        int   index    = dist(gen);
        auto& particle = particles[index];

        particle.id = id;

        new_particles.push_back(std::move(particle));
    }

    particles = std::move(new_particles);
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

    return particle;
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
