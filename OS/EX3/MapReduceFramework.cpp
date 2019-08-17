
// ======================= Imports =========================
#include "MapReduceFramework.h"
#include "Barrier.h"
#include  <atomic>
#include  <iostream>
#include  <queue>
#include  <semaphore.h>
#include <math.h>
#include <algorithm>

// ======================= Constants =========================
#define MAIN_THREAD 0


// ======================= Declarations =========================

struct JobContext;

/**
 * Struct containing all the information a thread needs to function in MapReduce.
 */
typedef struct Thread_Context
{
    int id;
    int number_of_threads;
    int* number_of_keys;
    std::atomic<unsigned long> *atomic_counter; //same as in the job
    std::vector<IntermediatePair>* *intermediate_vec; //array of intermediatePair
    std::vector<IntermediateVec *> *queue_for_shuffle;
    const InputVec *inputVec;
    const MapReduceClient *client;
    Barrier *my_barrier;
    pthread_mutex_t *queue_vec_mutex;
    sem_t *sem_for_queue;
    bool *shuffle_is_done;
    JobContext *my_job;
    OutputVec *my_output_vec;
    pthread_mutex_t *emit3_mutex;
    pthread_mutex_t *percentage_mutex;
    std::atomic<unsigned long> *atomic_counter_for_ruduce; //same as in the job

} Thread_Context;


/**
 * Jobs encompass all the threads and contain a status, and progress of the job that needs to be done,
 * as well as an array of all the threads that exist in the job including their context.
 */
typedef struct JobContext
{
    pthread_t *threads; //these are the relevant threads for the job ->int threads_num = multiThreadLevel;
    Thread_Context **thread_context_array = nullptr;
    JobState my_state = {UNDEFINED_STAGE, 0.0};
    bool have_we_waited = false;

} JobContext;

/**
 * the compare condition for the sort function.
 * @param l
 * @param r
 * @return boolean if left key is smaller than the right key
 */
bool compare_condition(const std::pair<K2 *, V2 *> &l, const std::pair<K2 *, V2 *> &r)
{
    return *l.first < *r.first;
}


/**
 * helper function which does the mapping phase, it includes the sort phase and introduces the barrier
 * at the end of the sort as well.
 * @param tc
 */
void map_for_thread(Thread_Context *tc)
{
    unsigned long old_value;
    int curr_id = tc->id;
    while (true)
    {

        old_value = (*(tc->atomic_counter))++;
        if (old_value >= tc->inputVec->size())
        {
            --(*(tc->atomic_counter));
            break;
        }
        tc->client->map(tc->inputVec->at(old_value).first, tc->inputVec->at(old_value).second, tc);

        //Will always reach 100%
        pthread_mutex_lock(tc->percentage_mutex);
        tc->my_job->my_state.percentage = float(*(tc->atomic_counter)) / tc->inputVec->size() * 100;
        pthread_mutex_unlock(tc->percentage_mutex);

    }
    std::sort(tc->intermediate_vec[curr_id]->begin(), tc->intermediate_vec[curr_id]->end(), compare_condition);

    tc->my_barrier->barrier();

}

/**
 * function which goes over all intermediate vectors and finds the max key out of them all.
 * @param tc (Thread Context)
 * @return
 */
K2 *find_max_key(std::vector<std::vector<IntermediatePair>>& copy_vec)
{
    K2 *cur_max_key = nullptr;
    for (long unsigned int i = 0; i < copy_vec.size(); i++)
    {
        if (i == 0)
        {
            cur_max_key = copy_vec[i].back().first;
        } else
        {
            K2 *temp_key = copy_vec[i].back().first;
            if (!(*temp_key < *cur_max_key))
            {
                cur_max_key = temp_key;
            }
        }
    }
    return cur_max_key;
}
bool is_empty_vec(std::vector<IntermediatePair>& a){

    return a.empty();
}
/**
 * Shuffle function accessed by the first thread, shuffles according to the key and adds them to the queue.
 * @param tc
 */
void shuffle_intermidiate(Thread_Context *tc)
{
    int total_keys = 0;
    for(int i = 0; i < tc->number_of_threads; i++)
    {
        total_keys += tc->intermediate_vec[i]->size(); //check this works
    }
    *tc->number_of_keys = total_keys;

    std::vector<std::vector<IntermediatePair>> copy_intermediate_pair;

    for(int i = 0; i<tc->number_of_threads; i++)
    {
        copy_intermediate_pair.push_back(*tc->intermediate_vec[i]); //TODO check if this is a pointer
    }
    copy_intermediate_pair.erase(std::remove_if(copy_intermediate_pair.begin(),copy_intermediate_pair.end(), is_empty_vec),copy_intermediate_pair.end()); //TODO make sure this doesnt fuck up shit


    while (!(copy_intermediate_pair.empty()))
    {
        auto *sequence = new IntermediateVec;
        K2 *max_key = find_max_key(copy_intermediate_pair);
        for (long unsigned int i = 0; i < copy_intermediate_pair.size(); i++)
        {
            while (!(*copy_intermediate_pair[i].back().first < *max_key) &&
                   !(*max_key < *copy_intermediate_pair[i].back().first))
            {
                sequence->push_back(copy_intermediate_pair[i].back());
                copy_intermediate_pair[i].pop_back();
                if (copy_intermediate_pair[i].empty())
                {
                    break;
                }
            }
        }

        pthread_mutex_lock(tc->queue_vec_mutex);
        (*tc->queue_for_shuffle).push_back(sequence);
        pthread_mutex_unlock(tc->queue_vec_mutex);


        sem_post(tc->sem_for_queue);
        copy_intermediate_pair.erase(std::remove_if(copy_intermediate_pair.begin(),copy_intermediate_pair.end(), is_empty_vec),copy_intermediate_pair.end()); //TODO make sure this works
    }
}

void reduce_values(Thread_Context *tc)
{

    while (true)
    {
        if (*tc->shuffle_is_done)
        {
            pthread_mutex_lock(tc->queue_vec_mutex);
            if ((*tc->queue_for_shuffle).empty())
            {
                pthread_mutex_unlock(tc->queue_vec_mutex);
                break;

            }
            pthread_mutex_unlock(tc->queue_vec_mutex);
        }

        sem_wait(tc->sem_for_queue);

        pthread_mutex_lock(tc->queue_vec_mutex); // must check if empty again just to make sure we dont try and access an empty queue.
        if ((*tc->queue_for_shuffle).empty())
        {
            pthread_mutex_unlock(tc->queue_vec_mutex);
            break;
        }

        IntermediateVec cur_sequence = *(*tc->queue_for_shuffle).back();
        //todo check if good
        delete ((*tc->queue_for_shuffle).back());
        (*tc->queue_for_shuffle).pop_back();

        pthread_mutex_unlock(tc->queue_vec_mutex);

        tc->client->reduce(&cur_sequence, tc);
        for (unsigned long i = 0; i<cur_sequence.size();i++)
        {
            (*tc->atomic_counter_for_ruduce)++;
        }
        pthread_mutex_lock(tc->percentage_mutex);
        tc->my_job->my_state.percentage = (float) (*tc->atomic_counter_for_ruduce) / *tc->number_of_keys * 100; //changed to total keys instead of inputvector size
        pthread_mutex_unlock(tc->percentage_mutex);
    }
}


/**
 * This is the function we pass on to the thread that does the algorithm for MapReduce.
 * @param arg
 * @return
 */
void *handler(void *arg)
{
    auto tc = (Thread_Context *) arg;
    if (!(tc->my_job->my_state.stage == MAP_STAGE))
    {
        tc->my_job->my_state.stage = MAP_STAGE;
    }
    map_for_thread(tc);

    if (tc->id == MAIN_THREAD)
    {
        tc->my_job->my_state = {REDUCE_STAGE, 0.0};
        shuffle_intermidiate(tc);
        //This frees every thread from the Semaphore waiting
        for (int i = 0; i < tc->number_of_threads; ++i)
        {
            sem_post(tc->sem_for_queue);
        }
        *tc->shuffle_is_done = true;
    }
    reduce_values(tc);
    pthread_exit(nullptr); // todo make sure we had a reson to get ireed of him

};


/**
 * init for the job
 */
JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
{

    int threads_num = multiThreadLevel;

    pthread_t *threads_v = new pthread_t[multiThreadLevel];
    Thread_Context **thread_context_array_t = new Thread_Context*[threads_num];
    //***********Init for the thread context*************
    std::atomic<unsigned long> *atomic_counter_job = new std::atomic<unsigned long>(0);
    std::atomic<unsigned long> *atomic_counter_job_for_reduce = new std::atomic<unsigned long>(0);

    Barrier *barrier_job = new Barrier(multiThreadLevel);
    pthread_mutex_t *queue_vex_mutex = new pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    pthread_mutex_t *emit3_mutex = new pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    pthread_mutex_t *percentage_mutex = new pthread_mutex_t(PTHREAD_MUTEX_INITIALIZER);
    int* num_of_keys = new int(0);
    sem_t *semaphore = new sem_t();
    sem_init(semaphore, 0, 0);
    //***********Init for the job*************
    JobContext *cur_job = new JobContext;
    cur_job->threads = threads_v;
    cur_job->thread_context_array = thread_context_array_t;
    //***************************************************
    std::vector<IntermediatePair>** intermidiate_vec = new std::vector<IntermediatePair>* [multiThreadLevel]; //to shuffle
    std::vector<IntermediateVec *> *queue = new std::vector<IntermediateVec *>;
    bool *was_shuffled = new bool(false);
    for (int i = 0; i < multiThreadLevel; i++)
    {
        // each slot in the array is the intermediate vector for the thread with ID i. 
        intermidiate_vec[i] = new std::vector<IntermediatePair>;
        thread_context_array_t[i] = (new Thread_Context{i, threads_num, num_of_keys, atomic_counter_job, intermidiate_vec, queue,
                                                        &inputVec, &client,
                                                        barrier_job, queue_vex_mutex, semaphore, was_shuffled, cur_job,
                                                        &outputVec, emit3_mutex, percentage_mutex,atomic_counter_job_for_reduce});
        pthread_create(cur_job->threads + i, nullptr, handler, thread_context_array_t[i]);

    }

    return cur_job; //pointer

}

/**
 * Deallocates all memory allocated by the startJobFramework
 * @param job
 */
void closeJobHandle(JobHandle job) //pointer
{

    waitForJob(job); // make sure wait for job was called atleast once if the user hasn't called it.
    JobContext *job_copy = (JobContext *) job;
    int multiThreadLevel = job_copy->thread_context_array[0]->number_of_threads;
    if (!(job_copy->my_state.stage == REDUCE_STAGE && (round(job_copy->my_state.percentage)) == 100.0))
    {
        return;
    } // just a sanity check

    delete job_copy->thread_context_array[0]->shuffle_is_done;
    delete job_copy->thread_context_array[0]->atomic_counter;
    delete job_copy->thread_context_array[0]->atomic_counter_for_ruduce;
    delete job_copy->thread_context_array[0]->my_barrier;
    delete job_copy->thread_context_array[0]->queue_vec_mutex;
    delete job_copy->thread_context_array[0]->percentage_mutex;
    delete job_copy->thread_context_array[0]->emit3_mutex;
    delete job_copy->thread_context_array[0]->sem_for_queue;
    delete job_copy->thread_context_array[0]->queue_for_shuffle;
    delete job_copy->thread_context_array[0]->number_of_keys;

    for (int i = 0; i < multiThreadLevel; i++)
    {
        delete job_copy->thread_context_array[0]->intermediate_vec[i]; //TODO make sure this works
    }

    delete[] job_copy->thread_context_array[0]->intermediate_vec;//should be empty

    for (int i = 0; i < multiThreadLevel; i++)
    {
        delete (job_copy->thread_context_array[i]);
    }

    delete[] job_copy->thread_context_array;
    delete[] job_copy->threads;
    delete job_copy;

    return;

}

/**
 * Adds a pair K2, V2 to the intermediate vector of that thread.
 * @param key
 * @param value
 * @param context
 */
void emit2(K2 *key, V2 *value, void *context)
{
    auto tc = (Thread_Context *) context;
    int curr_id = tc->id;
    tc->intermediate_vec[curr_id]->emplace_back(std::make_pair(key, value));
}


/**
 * Adds a pair for K3 , V3 to the output vector.
 * @param key
 * @param value
 * @param context
 */
void emit3(K3 *key, V3 *value, void *context)
{
    auto tc = (Thread_Context *) context;
    pthread_mutex_lock(tc->emit3_mutex);
    tc->my_output_vec->push_back(std::make_pair(key, value));
    pthread_mutex_unlock(tc->emit3_mutex);

}

/**
 * getter for the state.
 * @param job
 * @param state
 */
void getJobState(JobHandle job, JobState *state)
{
    JobContext *job_copy = (JobContext *) job;
    *state = job_copy->my_state;
}

/**
 * Function that joins threads together and waits for them to finish
 * for thread safety this function is only allowed to be called once per job.
 * @param job
 */
void waitForJob(JobHandle job)
{
    JobContext *job_copy = (JobContext *) job;
    if(job_copy->have_we_waited)
    {
        return; // we do not wait for the job twice!
    }
    job_copy->have_we_waited = true;
    for (int i = 0; i < job_copy->thread_context_array[0]->number_of_threads; i++)
    {
        if (pthread_join(job_copy->threads[i], nullptr) < 0)
        {
            exit(-1);
        }
    }
}
