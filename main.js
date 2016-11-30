  $(document).ready(function() {
    $('#contact_form').bootstrapValidator({
        // To use feedback icons, ensure that you use Bootstrap v3.1.0 or later
        feedbackIcons: {
            valid: 'glyphicon glyphicon-ok',
            invalid: 'glyphicon glyphicon-remove',
            validating: 'glyphicon glyphicon-refresh'
        },
        fields: {
            first_name: {
                validators: {
                        stringLength: {
                        min: 2,
                    },
                        notEmpty: {
                        message: 'Please supply your first name'
                    }
                }
            },
             last_name: {
                validators: {
                     stringLength: {
                        min: 2,
                    },
                    notEmpty: {
                        message: 'Please supply your last name'
                    }
                }
            },
            email: {
                validators: {
                    notEmpty: {
                        message: 'Please supply your email address'
                    },
                    emailAddress: {
                        message: 'Please supply a valid email address'
                    }
                }
            },
            phone: {
                validators: {
                    notEmpty: {
                        message: 'Please supply your phone number'
                    },
                    phone: {
                        country: 'US',
                        message: 'Please supply a vaild phone number with area code'
                    }
                }
            },
            address: {
                validators: {
                     stringLength: {
                        min: 8,
                    },
                    notEmpty: {
                        message: 'Please supply your street address'
                    }
                }
            },
            city: {
                validators: {
                     stringLength: {
                        min: 4,
                    },
                    notEmpty: {
                        message: 'Please supply your city'
                    }
                }
            },
            state: {
                validators: {
                    notEmpty: {
                        message: 'Please select your state'
                    }
                }
            },
            zip: {
                validators: {
                    notEmpty: {
                        message: 'Please supply your zip code'
                    },
                    zipCode: {
                        country: 'US',
                        message: 'Please supply a vaild zip code'
                    }
                }
            },
            comment: {
                validators: {
                      stringLength: {
                        min: 10,
                        max: 200,
                        message:'Please enter at least 10 characters and no more than 200'
                    },
                    notEmpty: {
                        message: 'Please supply a description of your project'
                    }
                    }
                }
            }
        })
        //.on('success.form.bv', function(e) {
            // $('#success_message').slideDown({ opacity: "show" }, "slow") // Do something ...
            //     $('#contact_form').data('bootstrapValidator').resetForm();

            // // Prevent form submission
            // e.preventDefault();

            // // Get the form instance
            // var $form = $(e.target);

            // // Get the BootstrapValidator instance
            // var bv = $form.data('bootstrapValidator');

            // Use Ajax to submit form data
            // $.post($form.attr('action'), $form.serialize(), function(result) {
            //     console.log(result);
            // }, 'json');
            //console.log('Success!');

       // });
});

var trained=0;
function getRadioCheckedValue(radio_name)
{
   var oRadio = document.thyroid_form.elements[radio_name];
 
   for(var i = 0; i < oRadio.length; i++)
   {
      if(oRadio[i].checked)
      {
         return oRadio[i].value;
      }
   }
 
   return '';
}

function Perceptron(input, hidden, output)
{
    // create the layers
    var inputLayer = new synaptic.Layer(input);
    var hiddenLayer = new synaptic.Layer(hidden);
    var outputLayer = new synaptic.Layer(output);

    // connect the layers
    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    inputLayer.set({
    squash: synaptic.Neuron.squash.TANH,
    //bias: 0
    })

    hiddenLayer.set({
    squash: synaptic.Neuron.squash.LOGISTIC,
    //bias: 0
    })

    // set the layers
    this.set({
        input: inputLayer,
        hidden: [hiddenLayer],
        output: outputLayer
    });
}

// extend the prototype chain
Perceptron.prototype = new synaptic.Network();
Perceptron.prototype.constructor = Perceptron;

function myFunction() {



features=["Age", "Sex", "On_thyroxine", "Query_on_thyroxine", "On_antithyroid_medication", "Sick", "Pregnant", "Thyroid Surgery", "I131_treatment", "Query_hypothyroid", "Query_hyperthyroid", "Lithium", "Goiter", "Tumor", "Hypopituitary", "Pysch", "TSH", "T3", "TT4", "T4U", "FTI"];

// prediction_input=[];
// prediction_input.push(document.thyroid_form.age.valueAsNumber/100);
// for (var j=1;j<16;j++){
//     prediction_input.push(parseInt(getRadioCheckedValue(features[j])));

// }
// prediction_input.push(document.thyroid_form.TSH.valueAsNumber/1000);
// prediction_input.push(document.thyroid_form.T3.valueAsNumber/10000);
// prediction_input.push(document.thyroid_form.TT4.valueAsNumber/100);
// prediction_input.push(document.thyroid_form.T4U.valueAsNumber/10);
// prediction_input.push(document.thyroid_form.FTI.valueAsNumber/100);

// console.log(prediction_input);

if (trained==0){

trainingSet=[];
testSet=[];
var Architect = synaptic.Architect;
var Trainer = synaptic.Trainer;
//var myPerceptron = new Architect.Perceptron(21,10,1);    
var myPerceptron = new Perceptron(21,3,1);
d3.csv("/data/UCI Thyroid Train_Anomaly.csv", function(data) {
    data.forEach(function(d) {
    var input_array=[];
    var output_array=[];
    for (var i=0;i<features.length;i++){        
       
        input_array.push(parseFloat(d[features[i]]));

    }
    output_array.push(parseInt(d["Class"]));
    dict={};
    dict['input']=input_array;
    dict['output']=output_array;
    trainingSet.push(dict);
    });

d3.csv("/data/UCI Thyroid Test_Anomaly.csv", function(data) {
    data.forEach(function(d) {
    var input_array=[];
    var output_array=[];
    for (var i=0;i<features.length;i++){        
       
        input_array.push(parseFloat(d[features[i]]));
    }
    output_array.push(parseInt(d["Class"]));
    dict={};
    dict['input']=input_array;
    dict['output']=output_array;
    testSet.push(dict);
    });


var trainer = new Trainer(myPerceptron);
trainer.train(trainingSet,{
    rate: .01,
    iterations: 20000,
    error: .005,
    shuffle: true,
    log: 1000,
    cost: Trainer.cost.CROSS_ENTROPY
});

var delay=10000; //10 second

setTimeout(function() {
  //your code to be executed after 1 second
  console.log("Wait Over")
  console.log(myPerceptron.activate(testSet[0]['input']));
  var j=0;
  for (var i=0;i<testSet.length;i++){
    if (myPerceptron.activate(testSet[i]['input'])<0.5){
        Out=0;
    }
    else{
        Out=1;
    }
    if (Out==testSet[i]['output'][0]){
        j=j+1;
    }
}
trained=1;
console.log(j/testSet.length);
//console.log(myPerceptron.activate(prediction_input));
console.log(myPerceptron.neurons());
}, delay);


    });    




});

}
else{

    //console.log(myPerceptron.activate(prediction_input));
}

}
function generateRandomBoolean(){

    if (Math.random()>0.5){
        return 1;
    }
    else{
        return 0;
    }
}

function fillBoolean(radio_name){
   var oRadio = document.thyroid_form.elements[radio_name];
 
   for(var i = 0; i < oRadio.length; i++)
   {
       oRadio[i].checked=generateRandomBoolean();
      
   }


}

function generateRandom(){
    
    document.thyroid_form.age.value=Math.ceil(Math.random()*100);
    document.thyroid_form.TSH.value=(Math.random()*50).toFixed(2);
    document.thyroid_form.T3.value=(Math.random()*400).toFixed(2);
    document.thyroid_form.TT4.value=(Math.random()*19).toFixed(2);
    document.thyroid_form.T4U.value=(Math.random()*20).toFixed(2);
    document.thyroid_form.FTI.value=(Math.random()*29).toFixed(2);
    features=["Age", "Sex", "On_thyroxine", "Query_on_thyroxine", "On_antithyroid_medication", "Sick", "Pregnant", "Thyroid Surgery", "I131_treatment", "Query_hypothyroid", "Query_hyperthyroid", "Lithium", "Goiter", "Tumor", "Hypopituitary", "Pysch", "TSH", "T3", "TT4", "T4U", "FTI"];
    for (var j=1;j<16;j++){
         fillBoolean(features[j]);

    }

}
