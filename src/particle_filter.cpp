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

#define SMALL 1e-4
#define LARGE 1e+10
#define NUMP  100

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if ( !is_initialized ) {

		default_random_engine gen;

    	num_particles = NUMP;

		// Create normal distributions
    	normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);


		for (int i = 0; i < num_particles; ++i) {
       		Particle p;
       		p.id = i;
	   		p.x  = dist_x(gen);
	   		p.y  = dist_y(gen);
	   		p.theta  = dist_theta(gen);	 
       		p.weight = 1.0;

       		particles.push_back(p);
        
		}

    	is_initialized=true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

  	// Create normal distributions
 	normal_distribution<double> dist_x(0, std_pos[0]);
  	normal_distribution<double> dist_y(0, std_pos[1]);
  	normal_distribution<double> dist_theta(0, std_pos[2]);

  	for (int i = 0; i < num_particles; i++) {
 
        Particle& p = particles[i];

    	// calculate new state
    	if (fabs(yaw_rate) < SMALL) {  
      		p.x += velocity * delta_t * cos(p.theta);
     		p.y += velocity * delta_t * sin(p.theta);
    	} else {
      	    p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      	    p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      	    p.theta += yaw_rate * delta_t;
    	}                                                                                                         
    	// add noise
    	p.x += dist_x(gen);
    	p.y += dist_y(gen);
    	p.theta += dist_theta(gen);
  	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


    for (int i = 0; i < observations.size(); i++) {

        LandmarkObs& o = observations[i];

        // find the predicted landmark nearest the target observed landmark
        double min_dist = LARGE;
        int min_id = -1;

        for (int j = 0; j < predicted.size(); j++) {
            LandmarkObs& p = predicted[j];
            double dist = sqrt((p.x-o.x)*(p.x-o.x) + (p.y-o.y)*(p.y-o.y));

            if (dist < min_dist) {
                min_dist = dist;
                min_id = p.id;
            }                                                                                                      
        }
        o.id = min_id;
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

    for (int i = 0; i < num_particles; i++) {

        Particle& p = particles[i];
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        // collect the (predicted) landmarks located within a sensor range from the target particle
        vector<LandmarkObs> landmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

            int  lm_id = map_landmarks.landmark_list[j].id_i;
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;
      
            if (sqrt(pow(lm_x-p_x,2) + pow(lm_y-p_y,2)) <= sensor_range) {
                landmarks.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }

        // estimate the weight
        double weight = 1.0;
        for (int j = 0; j < observations.size(); j++) {
            const LandmarkObs& o = observations[j];
             // transform the observation landmarks from the vehicle coordinates to map coordinates
            double t_x = cos(p_theta)*o.x - sin(p_theta)*o.y + p_x;
            double t_y = sin(p_theta)*o.x + cos(p_theta)*o.y + p_y;

            // It is better to apply directly instead of using the dataAssociation function !!!
            // get the x,y coordinate of the (predicted/nearest) landmark
            double lm_x, lm_y;
            double min_dist = LARGE;
            lm_x=0.0;
            lm_y=0.0;
            for (int k = 0; k < landmarks.size(); k++) {
                LandmarkObs& lmk=landmarks[k];
                double dist = sqrt(pow(lmk.x-t_x,2)+pow(lmk.y-t_y,2));
                if (dist<min_dist) {
                    min_dist = dist;
                    lm_x = lmk.x;
                    lm_y = lmk.y;
                }
            }

            //calculate weight with the multivariate Gaussian
            double s_x = std_landmark[0];
            double s_y = std_landmark[1];
            weight *= ( 1/(2*M_PI*s_x*s_y)) * 
                         exp( -(pow(lm_x-t_x,2)/(2*pow(s_x, 2)) 
                              +(pow(lm_y-t_y,2)/(2*pow(s_y, 2)))) );

        }
        particles[i].weight = weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
    default_random_engine gen;


    // get the weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    // generate random starting index [0, num_particles-1]
    uniform_int_distribution<int> intdist(0, num_particles-1);
    int index = intdist(gen);

    // get max weight
    double mw = *max_element(weights.begin(), weights.end());

    // uniform random distribution [0.0, max_weight]
    uniform_real_distribution<double> realdist(0.0, mw);

    double beta = 0.0;
    vector<Particle> t_particles;

    // spin the resample wheel!
    for (int i = 0; i < num_particles; i++) {
        beta += realdist(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        t_particles.push_back(particles[index]); 
    }

    particles = t_particles;

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

#undef NUMP
#undef LARGE
#undef SMALL

