const formTemp = document.querySelector('#formTemp');
const form = document.querySelector('form');
const formButton = document.querySelector('form button');
const countrySelect = document.querySelector('select#country');
const academicSelect = document.querySelector('select#academic');
const extracurricularSelect = document.querySelector('select#extracurricular');
const errorAlert = document.querySelector('#errorAlert');

const bookRec = document.querySelector('#bookRec ul');
const societyRec = document.querySelector('#societyRec ul');
const sportRec = document.querySelector('#sportRec ul');
const volunteerRec = document.querySelector('#volunteerRec ul');
const evalPerformance = document.querySelector('#eval ul');

const resetButton = document.querySelector('button[type=reset]');
const getEvalButton = document.querySelector('button#getEval');
const recommendationBox = document.querySelector('#recommendationBox');

const get_user_options = async () => {
    const options = await fetch('/user-options')
    if(options.ok) {
        formTemp.style.display = 'none';
        form.style.display = 'block';
    }
    const { data: { academic_activities, countries, extracurricular_activities } } = await options.json();

    countries.forEach((country, index) => {
        countrySelect.insertAdjacentHTML('beforeend', `<option value=${index+1}>${country}</option>`)
    })
    academic_activities.forEach((activity, index) => {
        academicSelect.insertAdjacentHTML('beforeend', `<option value=${index+1}>${activity}</option>`)
    })
    extracurricular_activities.forEach((activity, index) => {
        extracurricularSelect.insertAdjacentHTML('beforeend', `<option value=${index+1}>${activity}</option>`)
    })

    $('select#country').select2();
    $('select#academic').select2({
        maximumSelectionLength: 3
    });
    $('select#extracurricular').select2({
        maximumSelectionLength: 3
    });
    // options.json()
}

resetButton.addEventListener('click', function() {
    window.location.reload();
}, false);

const formatSerializeData = (data) => {
    $a = data.split('&');
    $result = {};
    $a.forEach(ele => {
        split = ele.split('=');
        if (Object($result).hasOwnProperty(split[0])) {
            $result[split[0]] += `,${split[1]}`;
        } else {
            $result[split[0]] = split[1];
        }
    })
    return $result;
}

const emptyResult = () => {
    $(bookRec).empty();
    $(societyRec).empty();
    $(sportRec).empty();
    $(volunteerRec).empty();
}

$('form').on('submit', async function(e) {
    e.preventDefault();
    const serializedData = $(this).serialize();
    const formData = formatSerializeData(serializedData);
    if(formData['country'] && formData['academic_interests'] && formData['extracurricular_interests']) {
        $(errorAlert).hide();
        
        $(formButton).attr('disabled', 'true').html(`
            <div class="spinner-border text-success" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        `);
        emptyResult();
        recommendationBox.scrollIntoView({ behavior: 'smooth' });
        const resp = await fetch('/recommend', {
            method: 'POST',
            body: JSON.stringify(formData),
            headers: {
                'content-type': 'application/json'
            },
            timeout: 5000
        })
        if(resp.ok) {
            const { 
                data: { books, societies, sports, volunteer_programs },
                links
            } = await resp.json();
            if(books.length) {
                books.forEach(book => {
                    bookRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">${book}</li>`)
                })
            } else {
                bookRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">No book recommendation</li>`)
            }
            
            if(societies.length) {
                societies.forEach((society, index) => {
                    societyRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">
                        ${society} - <a href="${links.societies[index]}" target="_blank">Learn more</a>
                    </li>`)
                })
            } else {
                societyRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">No society recommendation</li>`)
            }
            
            if(sports) {
                sports.forEach((sport, index) => {
                    sportRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">
                        ${sport} - <a href="${links.sports[index]}" target="_blank">Learn more</a>
                    </li>`)
                })
            } else {
                sportRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">No sport recommendation</li>`)
            }
            
            if(volunteer_programs) {
                volunteer_programs.forEach((volunteer, index) => {
                    volunteerRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">
                        ${volunteer}  - <a href="${links.volunteer_programs[index]}" target="_blank">Learn more</a>
                    </li>`)
                })
            } else {
                volunteerRec.insertAdjacentHTML('beforeend', `<li class="list-group-item">No volunteer program recommendation</li>`)
            }
            $(recommendationBox).show();

        } else {

        }
        $(formButton).removeAttr('disabled').text('Submit');
    } else {
        $(errorAlert).show();
    }
});

const get_model_evaluation = async () => {
    const options = await fetch('/evaluation')
    if(!options.ok) {
        // error getting evaluation
        return;
    }
    const { data: { overall_avg_precision, overall_avg_recall, overall_avg_f1 } } = await options.json();

    evalPerformance.insertAdjacentHTML('beforeend', `
        <li class="list-group-item">Overall Average Precision: ${(overall_avg_precision * 100).toFixed(1) + '%'}</li>
        <li class="list-group-item">Overall Average Recall: ${(overall_avg_recall * 100).toFixed(1) + '%'}</li>
        <li class="list-group-item">Overall Average F1-score: ${(overall_avg_f1 * 100).toFixed(1) + '%'}</li>
    `)
    $('#evalTemp').hide();
}

getEvalButton.addEventListener('click', function() {
    evalPerformance.innerHTML = '';
    $('#eval').show();
    $('#evalTemp').show();
    $('#eval')[0].scrollIntoView({ behavior: 'smooth' });
    get_model_evaluation();
}, false);

get_user_options();