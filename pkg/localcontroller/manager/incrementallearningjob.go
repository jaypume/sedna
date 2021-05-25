/*
Copyright 2021 The KubeEdge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package manager

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	"github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/storage"
	"github.com/kubeedge/sedna/pkg/localcontroller/trigger"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

// IncrementalLearningJob defines config for incremental-learning-job
type IncrementalLearningJob struct {
	sednav1.IncrementalLearningJob
	JobConfig *JobConfig
	Dataset   *Dataset
	Done      chan struct{}
	Storage   storage.Storage
}

// JobConfig defines config for incremental-learning-job
type JobConfig struct {
	UniqueIdentifier string
	Version          int
	Phase            string
	WorkerStatus     string
	TrainTrigger     trigger.Base
	DeployTrigger    trigger.Base
	TriggerStatus    string
	TriggerTime      time.Time
	TrainDataURL     string
	EvalDataURL      string
	OutputDir        string
	OutputConfig     *OutputConfig
	DataSamples      *DataSamples
	TrainModel       *TrainModel
	DeployModel      *ModelInfo
	EvalResult       []*ModelInfo
	Lock             sync.Mutex
}

// OutputConfig defines config for job output
type OutputConfig struct {
	SamplesOutput map[string]string `json:"trainData"`
	TrainOutput   string            `json:"trainOutput"`
	EvalOutput    string            `json:"evalOutput"`
}

// DataSamples defines samples information
type DataSamples struct {
	Numbers            int
	TrainSamples       []string
	EvalVersionSamples [][]string
	EvalSamples        []string
}

// TrainModel defines config about training model
type TrainModel struct {
	Model        *ModelInfo        `json:"model"`
	TrainedModel map[string]string `json:"trainedModel"`
	OutputURL    string            `json:"outputUrl"`
}

// IncrementalLearningJob defines incremental-learning-job manager
type IncrementalJobManager struct {
	Client               gmclient.ClientI
	WorkerMessageChannel chan WorkerMessage
	DatasetManager       *DatasetManager
	ModelManager         *ModelManager
	IncrementalJobMap    map[string]*IncrementalLearningJob
	VolumeMountPrefix    string
}

const (
	// JobIterationIntervalSeconds is interval time of each iteration of job
	JobIterationIntervalSeconds = 10
	// DatasetHandlerIntervalSeconds is interval time of handling dataset
	DatasetHandlerIntervalSeconds = 10
	// ModelHandlerIntervalSeconds is interval time of handling model
	ModelHandlerIntervalSeconds = 10
	// EvalSamplesCapacity is capacity of eval samples
	EvalSamplesCapacity = 5
	//IncrementalLearningJobKind is kind of incremental-learning-job resource
	IncrementalLearningJobKind = "incrementallearningjob"
)

// NewIncrementalJobManager creates a incremental-learning-job manager
func NewIncrementalJobManager(client gmclient.ClientI, datasetManager *DatasetManager,
	modelManager *ModelManager, options *options.LocalControllerOptions) *IncrementalJobManager {
	im := IncrementalJobManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
		DatasetManager:       datasetManager,
		ModelManager:         modelManager,
		IncrementalJobMap:    make(map[string]*IncrementalLearningJob),
		VolumeMountPrefix:    options.VolumeMountPrefix,
	}

	return &im
}

// Start starts incremental-learning-job manager
func (im *IncrementalJobManager) Start() error {
	go im.monitorWorker()

	return nil
}

// trainTask starts training task
func (im *IncrementalJobManager) trainTask(job *IncrementalLearningJob) error {
	jobConfig := job.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		payload, ok, err := im.triggerTrainTask(job)
		if !ok {
			return nil
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		err = im.Client.WriteMessage(payload, job.getHeader())
		if err != nil {
			klog.Errorf("job(name=%s) failed to write message: %v",
				jobConfig.UniqueIdentifier, err)
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	if jobConfig.WorkerStatus == WorkerFailedStatus {
		klog.Warningf("found the %sing phase worker that ran failed, "+
			"back the training phase triggering task", jobConfig.Phase)
		backTask(jobConfig)
	}

	if jobConfig.WorkerStatus == WorkerCompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)
		nextTask(jobConfig)
	}

	return nil
}

// evalTask starts eval task
func (im *IncrementalJobManager) evalTask(job *IncrementalLearningJob) error {
	jobConfig := job.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		payload, err := im.triggerEvalTask(job)
		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		err = im.Client.WriteMessage(payload, job.getHeader())
		if err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	if jobConfig.WorkerStatus == WorkerFailedStatus {
		msg := fmt.Sprintf("job(name=%s) found the %sing phase worker that ran failed, "+
			"back the training phase triggering task", jobConfig.UniqueIdentifier, jobConfig.Phase)
		klog.Errorf(msg)
		return fmt.Errorf(msg)
	}

	if jobConfig.WorkerStatus == WorkerCompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)
		nextTask(jobConfig)
	}

	return nil
}

// deployTask starts deploy task
func (im *IncrementalJobManager) deployTask(job *IncrementalLearningJob) error {
	jobConfig := job.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		neededDeploy, err := im.triggerDeployTask(job)
		status := UpstreamMessage{}
		if err == nil && neededDeploy {
			status.Phase = DeployPhase

			deployModel, err := im.deployModel(job)
			if err != nil {
				klog.Errorf("failed to deploy model for job(name=%s): %v", jobConfig.UniqueIdentifier, err)
			} else {
				klog.Infof("deployed model for job(name=%s) successfully", jobConfig.UniqueIdentifier)
			}
			if err != nil || deployModel == nil {
				status.Status = WorkerFailedStatus
			} else {
				status.Status = WorkerReadyStatus
				status.Input = &WorkerInput{
					Models: []ModelInfo{
						*deployModel,
					},
				}
			}
		} else {
			// No need to deploy, just report completed status
			// TODO: instead of reporting deploy-completed, another more reasonable status
			klog.Infof("no need to deploy model for job(name=%s)", jobConfig.UniqueIdentifier)
			status.Phase = DeployPhase
			status.Status = WorkerCompletedStatus
		}

		if err = im.Client.WriteMessage(status, job.getHeader()); err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	nextTask(jobConfig)

	klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)

	return nil
}

// startJob starts a job
func (im *IncrementalJobManager) startJob(name string) {
	var err error
	job := im.IncrementalJobMap[name]

	job.JobConfig = new(JobConfig)
	jobConfig := job.JobConfig
	jobConfig.UniqueIdentifier = name

	err = im.initJob(job)
	if err != nil {
		klog.Errorf("failed to init job (name=%s): %+v", jobConfig.UniqueIdentifier)
		return
	}

	klog.Infof("incremental job(name=%s) is started", name)
	defer klog.Infof("incremental learning job(name=%s) is stopped", name)
	go im.handleData(job)

	tick := time.NewTicker(JobIterationIntervalSeconds * time.Second)
	for {
		select {
		case <-job.Done:
			return
		default:
		}

		// in case model is not synced to LC before job synced to LC
		// here call loadModels in each period
		err := im.loadModels(job)
		if err != nil {
			klog.Warningf("job(name=%s) failed to load models, and waiting it: %v",
				jobConfig.UniqueIdentifier,
				err)
			<-tick.C
			continue
		}

		if job.Dataset == nil {
			klog.V(3).Infof("job(name=%s) dataset not ready",
				jobConfig.UniqueIdentifier)

			<-tick.C
			continue
		}

		switch jobConfig.Phase {
		case TrainPhase:
			err = im.trainTask(job)
		case EvalPhase:
			err = im.evalTask(job)
		case DeployPhase:
			err = im.deployTask(job)
		default:
			klog.Errorf("not vaild phase: %s", jobConfig.Phase)
			continue
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %s task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
		}

		<-tick.C
	}
}

// Insert inserts incremental-learning-job config to db
func (im *IncrementalJobManager) Insert(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	first := false
	job, ok := im.IncrementalJobMap[name]
	if !ok {
		job = &IncrementalLearningJob{}
		job.Storage = storage.Storage{IsLocalStorage: false}
		job.Done = make(chan struct{})
		im.IncrementalJobMap[name] = job
		first = true
	}

	if err := json.Unmarshal(message.Content, &job); err != nil {
		return err
	}

	credential := job.ObjectMeta.Annotations[CredentialAnnotationKey]
	if credential != "" {
		if err := job.Storage.SetCredential(credential); err != nil {
			return fmt.Errorf("failed to set job(name=%s)'s storage credential, error: %+v", name, err)
		}
	}

	if first {
		go im.startJob(name)
	}

	if err := db.SaveResource(name, job.TypeMeta, job.ObjectMeta, job.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes incremental-learning-job config in db
func (im *IncrementalJobManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if job, ok := im.IncrementalJobMap[name]; ok && job.Done != nil {
		close(job.Done)
	}

	delete(im.IncrementalJobMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// initJob inits the job object
func (im *IncrementalJobManager) initJob(job *IncrementalLearningJob) error {
	jobConfig := job.JobConfig
	jobConfig.TrainModel = new(TrainModel)
	jobConfig.TrainModel.OutputURL = jobConfig.OutputDir
	jobConfig.DeployModel = new(ModelInfo)
	jobConfig.Lock = sync.Mutex{}

	jobConfig.Version = 0
	jobConfig.Phase = TrainPhase
	jobConfig.WorkerStatus = WorkerReadyStatus
	jobConfig.TriggerStatus = TriggerReadyStatus
	trainTrigger, err := newTrigger(job.Spec.TrainSpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init train trigger: %+w", err)
	}
	deployTrigger, err := newTrigger(job.Spec.DeploySpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init deploy trigger: %+w", err)
	}
	jobConfig.TrainTrigger = trainTrigger
	jobConfig.DeployTrigger = deployTrigger

	outputDir := job.Spec.OutputDir

	isLocalURL, err := job.Storage.IsLocalURL(outputDir)
	if err != nil {
		return fmt.Errorf("job(name=%s)'s output dir is invalid, error: %+v", job.Name, outputDir)
	}

	if isLocalURL {
		job.Storage.IsLocalStorage = true
		outputDir = util.AddPrefixPath(im.VolumeMountPrefix, outputDir)
	}

	jobConfig.OutputDir = outputDir

	if err := job.createOutputDir(jobConfig); err != nil {
		return err
	}

	return nil
}

func newTrigger(t sednav1.Trigger) (trigger.Base, error) {
	// convert trigger to map
	triggerMap := make(map[string]interface{})
	c, err := json.Marshal(t)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(c, &triggerMap)
	if err != nil {
		return nil, err
	}
	return trigger.NewTrigger(triggerMap)
}

// triggerTrainTask triggers the train task
func (im *IncrementalJobManager) triggerTrainTask(job *IncrementalLearningJob) (interface{}, bool, error) {
	var err error
	jobConfig := job.JobConfig

	const numOfSamples = "num_of_samples"
	samples := map[string]interface{}{
		numOfSamples: len(jobConfig.DataSamples.TrainSamples),
	}

	isTrigger := jobConfig.TrainTrigger.Trigger(samples)

	if !isTrigger {
		return nil, false, nil
	}

	jobConfig.Version++

	var dataIndexURL string
	jobConfig.TrainDataURL, dataIndexURL, err = im.writeSamples(job, jobConfig.DataSamples.TrainSamples,
		jobConfig.OutputConfig.SamplesOutput["train"], jobConfig.Version, job.Dataset.Spec.Format, job.Dataset.URLPrefix)
	if err != nil {
		klog.Errorf("train phase: write samples to the file(%s) is failed, error: %v", jobConfig.TrainDataURL, err)
		return nil, false, err
	}

	format := jobConfig.TrainModel.Model.Format
	m := ModelInfo{
		Format: format,
		URL:    jobConfig.TrainModel.TrainedModel[format],
	}

	dataURL := jobConfig.TrainDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.TrainOutput, strconv.Itoa(jobConfig.Version)}, "/")
	if job.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(im.VolumeMountPrefix, outputDir)
	}

	input := WorkerInput{
		Models:       []ModelInfo{m},
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := UpstreamMessage{
		Phase:  TrainPhase,
		Status: WorkerReadyStatus,
		Input:  &input,
	}
	jobConfig.TriggerTime = time.Now()
	return &msg, true, nil
}

// triggerEvalTask triggers the eval task
func (im *IncrementalJobManager) triggerEvalTask(job *IncrementalLearningJob) (*UpstreamMessage, error) {
	jobConfig := job.JobConfig
	var err error

	var dataIndexURL string
	jobConfig.EvalDataURL, dataIndexURL, err = im.writeSamples(job, jobConfig.DataSamples.EvalSamples, jobConfig.OutputConfig.SamplesOutput["eval"],
		jobConfig.Version, job.Dataset.Spec.Format, job.Dataset.URLPrefix)
	if err != nil {
		klog.Errorf("job(name=%s) eval phase: write samples to the file(%s) is failed, error: %v",
			jobConfig.UniqueIdentifier, jobConfig.EvalDataURL, err)
		return nil, err
	}

	var models []ModelInfo
	models = append(models, ModelInfo{
		Format: "pb",
		URL:    jobConfig.TrainModel.TrainedModel["pb"],
	})

	models = append(models, ModelInfo{
		Format: jobConfig.DeployModel.Format,
		URL:    jobConfig.DeployModel.URL,
	})

	dataURL := jobConfig.EvalDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.EvalOutput, strconv.Itoa(jobConfig.Version)}, "/")
	if job.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(im.VolumeMountPrefix, outputDir)
	}

	input := WorkerInput{
		Models:       models,
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := &UpstreamMessage{
		Phase:  EvalPhase,
		Status: WorkerReadyStatus,
		Input:  &input,
	}

	return msg, nil
}

// triggerDeployTask triggers the deploy task
func (im *IncrementalJobManager) triggerDeployTask(job *IncrementalLearningJob) (bool, error) {
	jobConfig := job.JobConfig

	if len(jobConfig.EvalResult) != 2 {
		return false, fmt.Errorf("expected 2 evaluation results e, actual: %d", len(jobConfig.EvalResult))
	}

	newMetrics, oldMetrics := jobConfig.EvalResult[0].Metrics, jobConfig.EvalResult[1].Metrics
	metricDelta := make(map[string]interface{})

	for metric := range newMetrics {
		// keep the full metrics
		metricDelta[metric] = newMetrics[metric]
		var l []float64
		for i := range newMetrics[metric] {
			l = append(l, newMetrics[metric][i]-oldMetrics[metric][i])
		}
		metricDelta[metric+"_delta"] = l
	}
	tt := job.Spec.DeploySpec.Trigger

	// convert tt to map
	triggerMap := make(map[string]interface{})
	c, err := json.Marshal(tt)
	if err != nil {
		return false, err
	}

	err = json.Unmarshal(c, &triggerMap)
	if err != nil {
		return false, err
	}

	return jobConfig.DeployTrigger.Trigger(metricDelta), nil
}

// deployModel deploys model
func (im *IncrementalJobManager) deployModel(job *IncrementalLearningJob) (*ModelInfo, error) {
	jobConfig := job.JobConfig

	var models []ModelInfo
	for i := 0; i < len(jobConfig.EvalResult); i++ {
		models = append(models, ModelInfo{
			Format: jobConfig.EvalResult[i].Format,
			URL:    jobConfig.EvalResult[i].URL,
		})
	}

	var err error

	trainedModelFormat := models[0].Format
	deployModelFormat := job.JobConfig.DeployModel.Format
	if trainedModelFormat != deployModelFormat {
		return nil, fmt.Errorf("the trained model format(format=%s) is inconsistent with deploy model(format=%s)",
			deployModelFormat, deployModelFormat)
	}

	trainedModel := models[0].URL
	deployModel := job.JobConfig.DeployModel.URL
	if job.Storage.IsLocalStorage {
		trainedModel = util.AddPrefixPath(im.VolumeMountPrefix, trainedModel)
	}

	if err = job.updateDeployModel(deployModel, trainedModel); err != nil {
		return nil, err
	}

	klog.Infof("job(name=%s) deploys model(url=%s) successfully", jobConfig.UniqueIdentifier, trainedModel)

	return &models[0], nil
}

func (job *IncrementalLearningJob) updateDeployModel(deployModel string, newModel string) error {
	if err := job.Storage.CopyFile(newModel, deployModel); err != nil {
		return fmt.Errorf("copy model(url=%s) to the deploy model(url=%s) failed, error: %+v",
			newModel, deployModel, err)
	}

	klog.Infof("copy model(url=%s) to the deploy model(url=%s) successfully", newModel, deployModel)
	return nil
}

// createOutputDir creates the job output dir
func (job *IncrementalLearningJob) createOutputDir(jobConfig *JobConfig) error {
	outputDir := jobConfig.OutputDir

	dirNames := []string{"data/train", "data/eval", "train", "eval"}

	if job.Storage.IsLocalStorage {
		if err := util.CreateFolder(outputDir); err != nil {
			klog.Errorf("job(name=%s) create fold %s failed", jobConfig.UniqueIdentifier, outputDir)
			return err
		}

		for _, v := range dirNames {
			dir := path.Join(outputDir, v)
			if err := util.CreateFolder(dir); err != nil {
				klog.Errorf("job(name=%s) create fold %s failed", jobConfig.UniqueIdentifier, dir)
				return err
			}
		}
	}

	outputConfig := OutputConfig{
		SamplesOutput: map[string]string{
			"train": strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[0]}, "/"),
			"eval":  strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[1]}, "/"),
		},
		TrainOutput: strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[2]}, "/"),
		EvalOutput:  strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[3]}, "/"),
	}
	jobConfig.OutputConfig = &outputConfig

	return nil
}

// loadModels inits model information for training and deploying.
// it will be called by multi-times in case model not synced to LC before
// incremental job is synced to LC.
func (im *IncrementalJobManager) loadModels(job *IncrementalLearningJob) error {
	jobConfig := job.JobConfig
	getModel := func(name string) (sednav1.Model, error) {
		modelName := util.GetUniqueIdentifier(
			job.Namespace,
			name, ModelResourceKind)
		model, ok := im.ModelManager.GetModel(modelName)
		if !ok {
			return model, fmt.Errorf("not exists model(name=%s)", modelName)
		}
		return model, nil
	}

	if jobConfig.TrainModel.Model == nil {
		initialModel, err := getModel(job.Spec.InitialModel.Name)
		if err != nil {
			return err
		}

		jobConfig.TrainModel.Model = new(ModelInfo)
		jobConfig.TrainModel.TrainedModel = map[string]string{}
		format := initialModel.Spec.Format
		url := initialModel.Spec.URL
		jobConfig.TrainModel.Model.Format = format
		jobConfig.TrainModel.Model.URL = url
		jobConfig.TrainModel.TrainedModel[format] = url
	}

	if jobConfig.DeployModel != nil {
		evalModel, err := getModel(job.Spec.DeploySpec.Model.Name)
		if err != nil {
			return err
		}

		jobConfig.DeployModel = new(ModelInfo)
		jobConfig.DeployModel.Format = evalModel.Spec.Format
		jobConfig.DeployModel.URL = evalModel.Spec.URL
	}
	return nil
}

func (im *IncrementalJobManager) loadDataset(job *IncrementalLearningJob) error {
	if job.Dataset != nil {
		// already loaded
		return nil
	}

	datasetName := util.GetUniqueIdentifier(job.Namespace, job.Spec.Dataset.Name, DatasetResourceKind)
	dataset, ok := im.DatasetManager.GetDataset(datasetName)
	if !ok || dataset == nil {
		return fmt.Errorf("not exists dataset(name=%s)", datasetName)
	}

	jobConfig := job.JobConfig
	jobConfig.DataSamples = &DataSamples{
		Numbers:            0,
		TrainSamples:       make([]string, 0),
		EvalVersionSamples: make([][]string, 0),
		EvalSamples:        make([]string, 0),
	}

	job.Dataset = dataset
	return nil
}

// handleData updates samples information
func (im *IncrementalJobManager) handleData(job *IncrementalLearningJob) {
	tick := time.NewTicker(DatasetHandlerIntervalSeconds * time.Second)

	jobConfig := job.JobConfig
	iterCount := 0
	for {
		select {
		case <-job.Done:
			return
		default:
		}

		// in case dataset is not synced to LC before job synced to LC
		// here call loadDataset in each period
		err := im.loadDataset(job)
		if iterCount%100 == 0 {
			klog.Infof("job(name=%s) handling dataset", jobConfig.UniqueIdentifier)
		}
		iterCount++
		if err != nil {
			klog.Warningf("job(name=%s) failed to load dataset, and waiting it: %v",
				jobConfig.UniqueIdentifier,
				err)
			<-tick.C
			continue
		}

		dataset := job.Dataset

		if dataset.DataSource != nil && len(dataset.DataSource.TrainSamples) > jobConfig.DataSamples.Numbers {
			samples := dataset.DataSource.TrainSamples
			trainNum := int(job.Spec.Dataset.TrainProb * float64(len(samples)-jobConfig.DataSamples.Numbers))

			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = append(jobConfig.DataSamples.TrainSamples,
				samples[(jobConfig.DataSamples.Numbers+1):(jobConfig.DataSamples.Numbers+trainNum+1)]...)
			klog.Infof("job(name=%s) current train samples nums is %d",
				jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.TrainSamples))

			jobConfig.DataSamples.EvalVersionSamples = append(jobConfig.DataSamples.EvalVersionSamples,
				samples[(jobConfig.DataSamples.Numbers+trainNum+1):])
			jobConfig.Lock.Unlock()

			for _, v := range jobConfig.DataSamples.EvalVersionSamples {
				jobConfig.DataSamples.EvalSamples = append(jobConfig.DataSamples.EvalSamples, v...)
			}
			klog.Infof("job(name=%s) current eval samples nums is %d",
				jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.EvalSamples))

			jobConfig.DataSamples.Numbers = len(samples)
		}
		<-tick.C
	}
}

// createFile creates data file and data index file
func createFile(dir string, format string, isLocalStorage bool) (string, string) {
	switch format {
	case "txt":
		if isLocalStorage {
			return path.Join(dir, "data.txt"), ""
		}
		return strings.Join([]string{dir, "data.txt"}, "/"), strings.Join([]string{dir, "dataIndex.txt"}, "/")
	}
	return "", ""
}

// writeSamples writes samples information to a file
func (im *IncrementalJobManager) writeSamples(job *IncrementalLearningJob, samples []string, dir string, version int, format string, urlPrefix string) (string, string, error) {
	subDir := strings.Join([]string{dir, strconv.Itoa(version)}, "/")
	fileURL, absURLFile := createFile(subDir, format, job.Dataset.Storage.IsLocalStorage)

	if job.Storage.IsLocalStorage {
		if err := util.CreateFolder(subDir); err != nil {
			return "", "", err
		}
		if err := im.writeByLine(samples, fileURL); err != nil {
			return "", "", err
		}

		if !job.Dataset.Storage.IsLocalStorage {
			tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)
			if err := im.writeByLine(tempSamples, absURLFile); err != nil {
				return "", "", err
			}
		}

		return fileURL, absURLFile, nil
	}

	temporaryDir, err := util.CreateTemporaryDir()
	if err != nil {
		return "", "", err
	}

	localFileURL, localAbsURLFile := createFile(temporaryDir, format, job.Dataset.Storage.IsLocalStorage)

	if err := im.writeByLine(samples, localFileURL); err != nil {
		return "", "", err
	}

	if err := job.Storage.Upload(localFileURL, fileURL); err != nil {
		return "", "", err
	}

	tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)

	if err := im.writeByLine(tempSamples, localAbsURLFile); err != nil {
		return "", "", err
	}

	if err := job.Storage.Upload(localAbsURLFile, absURLFile); err != nil {
		return "", "", err
	}

	defer os.RemoveAll(localFileURL)
	defer os.RemoveAll(localAbsURLFile)

	return fileURL, absURLFile, nil
}

// writeByLine writes file by line
func (im *IncrementalJobManager) writeByLine(samples []string, fileURL string) error {
	file, err := os.Create(fileURL)
	if err != nil {
		klog.Errorf("create file(%s) failed", fileURL)
		return err
	}

	w := bufio.NewWriter(file)
	for _, line := range samples {
		_, _ = fmt.Fprintln(w, line)
	}
	if err := w.Flush(); err != nil {
		klog.Errorf("write file(%s) failed", fileURL)
		return err
	}

	if err := file.Close(); err != nil {
		klog.Errorf("close file failed, error: %v", err)
		return err
	}

	return nil
}

// monitorWorker monitors message from worker
func (im *IncrementalJobManager) monitorWorker() {
	for {
		workerMessageChannel := im.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}
		klog.V(4).Infof("handling worker message %+v", workerMessage)

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)

		job, ok := im.IncrementalJobMap[name]
		if !ok {
			continue
		}

		// TODO: filter some worker messages out
		wo := WorkerOutput{}
		wo.Models = workerMessage.Results
		wo.OwnerInfo = workerMessage.OwnerInfo

		msg := &UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &wo,
		}
		im.Client.WriteMessage(msg, job.getHeader())

		im.handleWorkerMessage(job, workerMessage)
	}
}

// handleWorkerMessage handles message from worker
func (im *IncrementalJobManager) handleWorkerMessage(job *IncrementalLearningJob, workerMessage WorkerMessage) {
	job.JobConfig.TrainModel.TrainedModel = make(map[string]string)

	jobPhase := job.JobConfig.Phase
	workerKind := workerMessage.Kind
	if jobPhase != workerKind {
		klog.Warningf("job(name=%s) %s phase get worker(kind=%s)", job.JobConfig.UniqueIdentifier,
			jobPhase, workerKind)
		return
	}

	var models []*ModelInfo
	for _, result := range workerMessage.Results {
		metrics := map[string][]float64{}
		if m, ok := result["metrics"]; ok {
			bytes, err := json.Marshal(m)
			if err != nil {
				return
			}

			err = json.Unmarshal(bytes, &metrics)
			if err != nil {
				klog.Warningf("failed to unmarshal the worker(name=%s) metrics %v, err: %v",
					workerMessage.Name,
					m,
					err)
			}
		}

		model := ModelInfo{
			result["format"].(string),
			result["url"].(string),
			metrics}
		models = append(models, &model)
	}

	job.JobConfig.WorkerStatus = workerMessage.Status

	if job.JobConfig.WorkerStatus == WorkerCompletedStatus {
		switch job.JobConfig.Phase {
		case TrainPhase:
			{
				for i := 0; i < len(models); i++ {
					format := models[i].Format
					if format != "" {
						job.JobConfig.TrainModel.TrainedModel[format] = models[i].URL
					}
				}
			}

		case EvalPhase:
			job.JobConfig.EvalResult = models
		}
	}
}

// forwardSamples deletes the samples information in the memory
func forwardSamples(jobConfig *JobConfig) {
	switch jobConfig.Phase {
	case TrainPhase:
		{
			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = jobConfig.DataSamples.TrainSamples[:0]
			jobConfig.Lock.Unlock()
		}
	case EvalPhase:
		{
			if len(jobConfig.DataSamples.EvalVersionSamples) > EvalSamplesCapacity {
				jobConfig.DataSamples.EvalVersionSamples = jobConfig.DataSamples.EvalVersionSamples[1:]
			}
		}
	}
}

// backTask backs train task status
func backTask(jobConfig *JobConfig) {
	jobConfig.Phase = TrainPhase
	initTaskStatus(jobConfig)
}

// initTaskStatus inits task status
func initTaskStatus(jobConfig *JobConfig) {
	jobConfig.WorkerStatus = WorkerReadyStatus
	jobConfig.TriggerStatus = TriggerReadyStatus
}

// nextTask converts next task status
func nextTask(jobConfig *JobConfig) {
	switch jobConfig.Phase {
	case TrainPhase:
		{
			forwardSamples(jobConfig)
			initTaskStatus(jobConfig)
			jobConfig.Phase = EvalPhase
		}

	case EvalPhase:
		{
			forwardSamples(jobConfig)
			initTaskStatus(jobConfig)
			jobConfig.Phase = DeployPhase
		}
	case DeployPhase:
		{
			backTask(jobConfig)
		}
	}
}

// AddWorkerMessage adds worker messages
func (im *IncrementalJobManager) AddWorkerMessage(message WorkerMessage) {
	im.WorkerMessageChannel <- message
}

// GetName returns name of the manager
func (im *IncrementalJobManager) GetName() string {
	return IncrementalLearningJobKind
}

func (job *IncrementalLearningJob) getHeader() gmclient.MessageHeader {
	return gmclient.MessageHeader{
		Namespace:    job.Namespace,
		ResourceKind: job.Kind,
		ResourceName: job.Name,
		Operation:    gmclient.StatusOperation,
	}
}
