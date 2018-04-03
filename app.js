var tf = require('@tensorflow/tfjs');
var express = require('express');
var bodyParser = require('body-parser');
var cors = require('cors');

var app = express();

app.use(bodyParser.json());
app.use(cors());
app.use(bodyParser.urlencoded({
    extended: false
}));

app.get('/', (req, res) => {
    res.send("hello");
});

function predict(x, a, b, c) {
    // ax^2 + bx + c
    console.log("called");
    return tf.tidy(() => {
        const input = tf.scalar(x);
        const ax2 = a.mul(input.square());
        const bx = b.mul(input);
        const y = ax2.add(bx).add(c);

        console.log(y);
        return y;
    });
}

var calculate = async function (range, a, b, c) {
    var y_arr = [];
    for (var i = 0; i < range + 1; i++) {
        var y = predict(i, a, b, c);
        var value = await y.data();
        y_arr.push(value[0]);
    }

    return y_arr;
}

app.post('/calculate', async function (req, res) {
    const a = tf.scalar(parseInt(req.body.a));
    const b = tf.scalar(parseInt(req.body.b));
    const c = tf.scalar(parseInt(req.body.c));
    const range = parseInt(req.body.range);

    var x_arr = [];
    var y_arr = [];
    for (var i = 0; i < range + 1; i++) {
        x_arr.push(i);
    }

    var call = calculate(range, a, b, c).then((returnvar) => {
        y_arr = returnvar;
        res.status(200).json({
            x_arr: x_arr,
            y_arr: y_arr
        });
    }).catch((err) => {
        res.status(503);
    });

});


app.listen(3000);